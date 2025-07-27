import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

/**
 * AI-powered Git Impact Analyzer using LLMs for semantic code analysis
 * Superior to static analysis - understands context, semantics, and complex relationships
 */
export class AIGitImpactAnalyzer {
    constructor(projectPath, options = {}) {
        this.projectPath = projectPath;
        this.options = {
            llmProvider: options.llmProvider || 'gemini', // openai, anthropic, local, gemini, ollama
            model: options.model || 'gemini-1.5-pro',
            maxContextFiles: options.maxContextFiles || 10,
            analysisDepth: options.analysisDepth || 'deep', // shallow, medium, deep

            // Batching and rate limiting options
            batchSize: options.batchSize || 5, // Files per batch
            maxTokensPerBatch: options.maxTokensPerBatch || 8000, // Token limit per batch
            rateLimitDelay: options.rateLimitDelay || 1000, // ms between requests
            maxRetries: options.maxRetries || 3,
            retryDelay: options.retryDelay || 2000, // Base retry delay in ms

            ...options
        };

        // Rate limiting state
        this.lastRequestTime = 0;
    }

    /**
     * Main method: Analyze git diff impact using AI with intelligent batching
     */
    async analyzeGitDiffWithAI() {
        console.log('🤖 Starting AI-powered git diff impact analysis with batching...');

        try {
            // Step 1: Get git diff
            const gitDiff = await this.getGitDiff();
            if (!gitDiff.trim()) {
                return { message: 'No changes detected in git diff', impacts: [] };
            }

            // Step 2: Extract changed files and functions
            const changes = await this.parseGitDiff(gitDiff);

            // Step 3: Get relevant context files
            const contextFiles = await this.gatherContextFiles(changes);

            // Step 4: Perform batched AI analysis
            const impactAnalysis = await this.performBatchedAIAnalysis(changes, contextFiles, gitDiff);

            return {
                changedFiles: changes.files.length,
                impactAnalysis,
                confidence: impactAnalysis.confidence || 0.85,
                recommendations: impactAnalysis.recommendations || []
            };

        } catch (error) {
            console.error('❌ AI Impact Analysis failed:', error.message);
            return { error: error.message, impacts: [] };
        }
    }

    /**
     * Perform AI analysis with intelligent batching to avoid rate limits
     */
    async performBatchedAIAnalysis(changes, contextFiles, gitDiff) {
        console.log(`📦 Batching analysis for ${contextFiles.length} files...`);

        // Create batches based on size and token limits
        const batches = this.createIntelligentBatches(changes, contextFiles, gitDiff);
        console.log(`🔄 Created ${batches.length} batches for analysis`);

        const batchResults = [];

        // Process batches with rate limiting
        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            console.log(`⏳ Processing batch ${i + 1}/${batches.length} (${batch.files.length} files)...`);

            try {
                // Apply rate limiting
                await this.applyRateLimit();

                // Analyze batch with retry logic
                const batchResult = await this.analyzeBatchWithRetry(batch, i + 1);
                batchResults.push(batchResult);

                console.log(`✅ Batch ${i + 1} completed successfully`);

            } catch (error) {
                console.error(`❌ Batch ${i + 1} failed:`, error.message);
                // Continue with other batches, don't fail completely
                batchResults.push(this.getFallbackBatchAnalysis(batch));
            }
        }

        // Merge batch results into comprehensive analysis
        return this.mergeBatchResults(batchResults, changes);
    }

    /**
     * Create intelligent batches based on file size and token limits
     */
    createIntelligentBatches(changes, contextFiles, gitDiff) {
        const batches = [];
        let currentBatch = {
            changes: { ...changes, files: [] },
            files: [],
            gitDiff: gitDiff,
            estimatedTokens: 0
        };

        // Add core git diff to first batch
        currentBatch.estimatedTokens += this.estimateTokens(gitDiff);

        // Group files into batches
        for (const file of contextFiles) {
            const fileTokens = this.estimateTokens(file.content);

            // If adding this file would exceed limits, start new batch
            if ((currentBatch.files.length >= this.options.batchSize ||
                currentBatch.estimatedTokens + fileTokens > this.options.maxTokensPerBatch) &&
                currentBatch.files.length > 0) {

                batches.push(currentBatch);
                currentBatch = {
                    changes: { ...changes, files: [] },
                    files: [],
                    gitDiff: gitDiff,
                    estimatedTokens: this.estimateTokens(gitDiff)
                };
            }

            // Add file to current batch
            currentBatch.files.push(file);
            currentBatch.changes.files.push(file.path);
            currentBatch.estimatedTokens += fileTokens;
        }

        // Add final batch if not empty
        if (currentBatch.files.length > 0) {
            batches.push(currentBatch);
        }

        // If no batches created, create a minimal one
        if (batches.length === 0) {
            batches.push({
                changes,
                files: contextFiles.slice(0, 3), // Minimal set
                gitDiff,
                estimatedTokens: this.estimateTokens(gitDiff)
            });
        }

        return batches;
    }

    /**
     * Estimate token count for content (rough approximation)
     */
    estimateTokens(content) {
        if (!content) return 0;
        // Rough estimate: ~4 characters per token
        return Math.ceil(content.length / 4);
    }

    /**
     * Apply rate limiting between requests
     */
    async applyRateLimit() {
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;

        if (timeSinceLastRequest < this.options.rateLimitDelay) {
            const waitTime = this.options.rateLimitDelay - timeSinceLastRequest;
            console.log(`⏱️  Rate limiting: waiting ${waitTime}ms...`);
            await this.sleep(waitTime);
        }

        this.lastRequestTime = Date.now();
    }

    /**
     * Analyze a single batch with retry logic
     */
    async analyzeBatchWithRetry(batch, batchNumber) {
        let lastError;

        for (let attempt = 1; attempt <= this.options.maxRetries; attempt++) {
            try {
                console.log(`🔄 Batch ${batchNumber}, attempt ${attempt}/${this.options.maxRetries}`);

                const prompt = this.buildBatchAnalysisPrompt(batch);
                const result = await this.callLLMAPI(prompt);

                // Validate result structure
                if (this.validateAnalysisResult(result)) {
                    return { ...result, batchNumber, success: true };
                } else {
                    throw new Error('Invalid analysis result structure');
                }

            } catch (error) {
                lastError = error;

                if (attempt < this.options.maxRetries) {
                    const delay = this.options.retryDelay * Math.pow(2, attempt - 1); // Exponential backoff
                    console.log(`⚠️  Batch ${batchNumber} attempt ${attempt} failed: ${error.message}`);
                    console.log(`🔄 Retrying in ${delay}ms...`);
                    await this.sleep(delay);
                } else {
                    console.error(`❌ Batch ${batchNumber} failed after ${this.options.maxRetries} attempts`);
                }
            }
        }

        throw lastError;
    }

    /**
     * Build optimized prompt for batch analysis
     */
    buildBatchAnalysisPrompt(batch) {
        const { changes, files, gitDiff } = batch;

        // Create more concise prompt for batch processing
        return `
You are an expert code analyst. Analyze this git diff batch and determine the impact of changes.

## Batch Info:
- Files in batch: ${files.length}
- Estimated tokens: ${batch.estimatedTokens}

## Changed Files in Batch:
${changes.files.map(f => `- ${f}`).join('\n')}

## Git Diff (relevant portions):
\`\`\`diff
${gitDiff.slice(0, 3000)}${gitDiff.length > 3000 ? '\n...[truncated for batch processing]' : ''}
\`\`\`

## Context Files (batch):
${files.slice(0, 5).map(f => `### ${f.path}:\n\`\`\`${path.extname(f.path).slice(1)}\n${f.content.slice(0, 1000)}${f.content.length > 1000 ? '\n...[truncated]' : ''}\n\`\`\`\n`).join('')}

Provide concise analysis focusing on:
1. Direct impact within this batch
2. Risk level for this batch
3. Key testing recommendations
4. Critical dependencies

Respond with valid JSON:
{
  "impactedFiles": ["file1", "file2"],
  "impactedFunctions": ["func1()", "func2()"],
  "riskLevel": "low|medium|high",
  "testingRecommendations": ["test1", "test2"],
  "deploymentNotes": ["note1", "note2"], 
  "confidence": 0.8,
  "explanation": "Brief explanation",
  "batchSummary": "Summary of this batch impact"
}`;
    }

    /**
     * Validate analysis result structure
     */
    validateAnalysisResult(result) {
        if (!result || typeof result !== 'object') return false;

        const requiredFields = ['impactedFiles', 'impactedFunctions', 'riskLevel', 'confidence'];
        return requiredFields.every(field => result.hasOwnProperty(field));
    }

    /**
     * Merge results from all batches into comprehensive analysis
     */
    mergeBatchResults(batchResults, originalChanges) {
        const merged = {
            impactedFiles: new Set(),
            impactedFunctions: new Set(),
            testingRecommendations: new Set(),
            deploymentNotes: new Set(),
            explanations: [],
            riskLevels: [],
            confidences: []
        };

        // Aggregate all batch results
        for (const batch of batchResults) {
            if (batch.success && batch.impactedFiles) {
                batch.impactedFiles.forEach(f => merged.impactedFiles.add(f));
                batch.impactedFunctions.forEach(f => merged.impactedFunctions.add(f));

                if (batch.testingRecommendations) {
                    batch.testingRecommendations.forEach(t => merged.testingRecommendations.add(t));
                }

                if (batch.deploymentNotes) {
                    batch.deploymentNotes.forEach(d => merged.deploymentNotes.add(d));
                }

                if (batch.explanation) merged.explanations.push(batch.explanation);
                if (batch.riskLevel) merged.riskLevels.push(batch.riskLevel);
                if (batch.confidence) merged.confidences.push(batch.confidence);
            }
        }

        // Calculate overall risk level
        const riskPriority = { high: 3, medium: 2, low: 1 };
        const maxRisk = merged.riskLevels.reduce((max, risk) =>
            riskPriority[risk] > riskPriority[max] ? risk : max, 'low'
        );

        // Calculate average confidence
        const avgConfidence = merged.confidences.length > 0
            ? merged.confidences.reduce((sum, c) => sum + c, 0) / merged.confidences.length
            : 0.5;

        return {
            impactedFiles: Array.from(merged.impactedFiles),
            impactedFunctions: Array.from(merged.impactedFunctions),
            riskLevel: maxRisk,
            testingRecommendations: Array.from(merged.testingRecommendations),
            deploymentNotes: Array.from(merged.deploymentNotes),
            confidence: avgConfidence,
            explanation: `Batched analysis of ${batchResults.length} batches. ${merged.explanations.join(' ')}`,
            batchCount: batchResults.length,
            successfulBatches: batchResults.filter(b => b.success).length
        };
    }

    /**
     * Get fallback analysis for failed batch
     */
    getFallbackBatchAnalysis(batch) {
        return {
            impactedFiles: batch.changes.files,
            impactedFunctions: ['[Analysis failed - manual review needed]'],
            riskLevel: 'medium',
            testingRecommendations: ['Comprehensive testing recommended due to analysis failure'],
            deploymentNotes: ['Extra caution - batch analysis failed'],
            confidence: 0.1,
            explanation: `Batch analysis failed for ${batch.files.length} files`,
            success: false
        };
    }

    /**
     * Sleep utility for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get git diff output
     */
    async getGitDiff() {
        return new Promise((resolve, reject) => {
            const gitProcess = spawn('git', ['diff', '--unified=3'], {
                stdio: 'pipe',
                cwd: this.projectPath
            });

            let output = '';
            let errorOutput = '';

            gitProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            gitProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            gitProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(output);
                } else {
                    // Try git status if no diff available
                    this.getGitStatus().then(resolve).catch(reject);
                }
            });
        });
    }

    /**
     * Parse git diff to extract changes
     */
    async parseGitDiff(gitDiff) {
        const files = [];
        const functions = [];
        const variables = [];

        const lines = gitDiff.split('\n');
        let currentFile = null;

        for (const line of lines) {
            if (line.startsWith('diff --git')) {
                const match = line.match(/diff --git a\/(.*?) b\/(.*)/);
                if (match) {
                    currentFile = match[1];
                    files.push(currentFile);
                }
            } else if (line.startsWith('+') && !line.startsWith('+++')) {
                // Extract added/modified content for AI analysis
                const addedLine = line.substring(1).trim();
                if (addedLine.includes('function ') || addedLine.includes('const ') || addedLine.includes('let ')) {
                    functions.push({ file: currentFile, line: addedLine, type: 'added' });
                }
            } else if (line.startsWith('-') && !line.startsWith('---')) {
                // Extract removed content
                const removedLine = line.substring(1).trim();
                if (removedLine.includes('function ') || removedLine.includes('const ') || removedLine.includes('let ')) {
                    functions.push({ file: currentFile, line: removedLine, type: 'removed' });
                }
            }
        }

        return { files, functions, variables, rawDiff: gitDiff };
    }

    /**
     * Gather relevant context files for AI analysis
     */
    async gatherContextFiles(changes) {
        const contextFiles = [];

        // Get changed files content
        for (const file of changes.files) {
            try {
                const filePath = path.join(this.projectPath, file);
                const content = await fs.readFile(filePath, 'utf-8');
                contextFiles.push({
                    path: file,
                    content: content.slice(0, 10000), // Limit content size
                    type: 'changed'
                });
            } catch (error) {
                console.warn(`⚠️ Could not read file: ${file}`);
            }
        }

        // Get related files based on imports/requires
        const relatedFiles = await this.findRelatedFiles(changes.files);
        for (const file of relatedFiles.slice(0, this.options.maxContextFiles)) {
            try {
                const content = await fs.readFile(file, 'utf-8');
                contextFiles.push({
                    path: path.relative(this.projectPath, file),
                    content: content.slice(0, 5000), // Smaller size for context files
                    type: 'related'
                });
            } catch (error) {
                console.warn(`⚠️ Could not read related file: ${file}`);
            }
        }

        return contextFiles;
    }

    /**
     * Find files related to the changed files through imports/requires
     */
    async findRelatedFiles(changedFiles) {
        const related = new Set();

        for (const file of changedFiles) {
            try {
                const filePath = path.join(this.projectPath, file);
                const content = await fs.readFile(filePath, 'utf-8');

                // Extract import/require statements
                const importRegex = /(?:import.*from\s+['"`]([^'"`]+)['"`]|require\s*\(\s*['"`]([^'"`]+)['"`]\))/g;
                let match;

                while ((match = importRegex.exec(content)) !== null) {
                    const importPath = match[1] || match[2];
                    if (importPath && !importPath.startsWith('.')) {
                        // Skip node_modules
                        continue;
                    }

                    // Resolve relative import paths
                    const resolvedPath = path.resolve(path.dirname(filePath), importPath);
                    const extensions = ['.js', '.ts', '.jsx', '.tsx'];

                    for (const ext of extensions) {
                        const fullPath = resolvedPath + ext;
                        try {
                            await fs.access(fullPath);
                            related.add(fullPath);
                            break;
                        } catch {
                            // Try next extension
                        }
                    }
                }
            } catch (error) {
                console.warn(`⚠️ Error finding related files for: ${file}`);
            }
        }

        return Array.from(related);
    }

    /**
     * Perform AI analysis using LLM
     */
    async performAIAnalysis(changes, contextFiles, gitDiff) {
        const prompt = this.buildAnalysisPrompt(changes, contextFiles, gitDiff);

        // This would integrate with your preferred LLM API
        // For now, returning a structured analysis format
        return await this.callLLMAPI(prompt);
    }

    /**
     * Build comprehensive prompt for LLM analysis
     */
    buildAnalysisPrompt(changes, contextFiles, gitDiff) {
        return `
You are an expert code analyst. Analyze this git diff and determine the impact of changes across the codebase.

## Changed Files:
${changes.files.map(f => `- ${f}`).join('\n')}

## Git Diff:
\`\`\`diff
${gitDiff.slice(0, 5000)} ${gitDiff.length > 5000 ? '...[truncated]' : ''}
\`\`\`

## Context Files:
${contextFiles.map(f => `
### ${f.path} (${f.type}):
\`\`\`${path.extname(f.path).slice(1)}
${f.content.slice(0, 2000)}${f.content.length > 2000 ? '\n...[truncated]' : ''}
\`\`\`
`).join('\n')}

## Analysis Required:
1. **Direct Impact**: What functions/APIs are directly affected by these changes?
2. **Indirect Impact**: What other parts of the system might be affected through dependencies?
3. **Risk Assessment**: What's the risk level of these changes? (Low/Medium/High)
4. **Testing Recommendations**: What should be tested based on these changes?
5. **Deployment Considerations**: Any special considerations for deployment?

Please provide a structured JSON response with:
- impactedFiles: Array of files that might be affected
- impactedFunctions: Array of functions that might be affected  
- riskLevel: "low" | "medium" | "high"
- testingRecommendations: Array of testing suggestions
- deploymentNotes: Array of deployment considerations
- confidence: Number between 0-1 indicating analysis confidence
- explanation: Detailed explanation of the analysis

Format your response as valid JSON.
`;
    }

    /**
     * Call LLM API (now with Gemini and Ollama support!)
     */
    async callLLMAPI(prompt) {
        console.log(`🤖 Sending prompt to ${this.options.llmProvider}...`);

        try {
            switch (this.options.llmProvider) {
                case 'ollama':
                    return await this.callOllamaAPI(prompt);
                case 'gemini':
                    return await this.callGeminiAPI(prompt);
                case 'openai':
                    return await this.callOpenAIAPI(prompt);
                case 'anthropic':
                    return await this.callAnthropicAPI(prompt);
                default:
                    throw new Error(`Unsupported LLM provider: ${this.options.llmProvider}`);
            }
        } catch (error) {
            console.error('❌ LLM API call failed:', error.message);
            // Return fallback analysis
            return this.getFallbackAnalysis();
        }
    }

    /**
     * Ollama local model integration
     */
    async callOllamaAPI(prompt) {
        const ollamaUrl = this.options.ollamaUrl || 'http://localhost:11434';
        const model = this.options.model || 'deepseek-coder:6.7b';

        console.log(`🏠 Using local Ollama model: ${model}`);

        try {
            const response = await fetch(`${ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model,
                    prompt: `${prompt}\n\nIMPORTANT: You must respond with valid JSON only. No markdown, no explanations outside the JSON. The response should be a single JSON object with the exact structure requested.`,
                    stream: false,
                    options: {
                        temperature: 0.1,
                        top_p: 0.9,
                        top_k: 40,
                        repeat_penalty: 1.1,
                        num_predict: 4096
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            const text = data.response;

            try {
                // Clean the response to extract JSON
                const jsonMatch = text.match(/\{[\s\S]*\}/);
                if (!jsonMatch) {
                    // Try to find JSON in the response more aggressively
                    const lines = text.split('\n');
                    for (const line of lines) {
                        if (line.trim().startsWith('{')) {
                            const possibleJson = line.trim();
                            try {
                                return JSON.parse(possibleJson);
                            } catch (e) {
                                continue;
                            }
                        }
                    }
                    throw new Error('No valid JSON found in Ollama response');
                }

                return JSON.parse(jsonMatch[0]);
            } catch (parseError) {
                console.error('❌ Failed to parse Ollama response as JSON:', parseError.message);
                console.error('Raw response:', text);

                // Try to extract basic info from text response
                return this.parseTextResponseToJSON(text);
            }

        } catch (error) {
            console.error('❌ Ollama API call failed:', error.message);
            throw error;
        }
    }

    /**
     * Parse text response to JSON when model doesn't return proper JSON
     */
    parseTextResponseToJSON(text) {
        console.log('🔧 Attempting to parse text response to JSON...');

        // Extract basic information from text
        const impactedFiles = [];
        const impactedFunctions = [];
        const testingRecommendations = [];
        const deploymentNotes = [];

        // Simple regex patterns to extract information
        const fileMatches = text.match(/(?:file|impact|affect)(?:ed|s)?:?\s*([^\n]+)/gi);
        const functionMatches = text.match(/function(?:s)?:?\s*([^\n]+)/gi);
        const testMatches = text.match(/(?:test|testing):?\s*([^\n]+)/gi);
        const deployMatches = text.match(/(?:deploy|deployment):?\s*([^\n]+)/gi);

        if (fileMatches) {
            fileMatches.forEach(match => {
                const files = match.split(/[,\s]+/).filter(f => f.includes('.'));
                impactedFiles.push(...files);
            });
        }

        if (functionMatches) {
            functionMatches.forEach(match => {
                const funcs = match.split(/[,\s]+/).filter(f => f.includes('('));
                impactedFunctions.push(...funcs);
            });
        }

        if (testMatches) {
            testingRecommendations.push(...testMatches.map(m => m.replace(/(?:test|testing):?\s*/i, '').trim()));
        }

        if (deployMatches) {
            deploymentNotes.push(...deployMatches.map(m => m.replace(/(?:deploy|deployment):?\s*/i, '').trim()));
        }

        // Determine risk level from text
        let riskLevel = 'medium';
        if (text.toLowerCase().includes('high risk') || text.toLowerCase().includes('critical')) {
            riskLevel = 'high';
        } else if (text.toLowerCase().includes('low risk') || text.toLowerCase().includes('minimal')) {
            riskLevel = 'low';
        }

        return {
            impactedFiles: impactedFiles.slice(0, 10) || ['Unable to extract from text response'],
            impactedFunctions: impactedFunctions.slice(0, 10) || ['Unable to extract from text response'],
            riskLevel,
            testingRecommendations: testingRecommendations.slice(0, 5) || ['Comprehensive testing recommended'],
            deploymentNotes: deploymentNotes.slice(0, 5) || ['Monitor deployment carefully'],
            confidence: 0.6, // Lower confidence for parsed text
            explanation: `Parsed from text response: ${text.slice(0, 200)}...`
        };
    }

    /**
     * Check if Ollama is running and model is available
     */
    async checkOllamaStatus() {
        const ollamaUrl = this.options.ollamaUrl || 'http://localhost:11434';
        const model = this.options.model || 'deepseek-coder:6.7b';

        try {
            // Check if Ollama is running
            const response = await fetch(`${ollamaUrl}/api/tags`);
            if (!response.ok) {
                throw new Error('Ollama not running');
            }

            const data = await response.json();
            const availableModels = data.models.map(m => m.name);

            if (!availableModels.includes(model)) {
                console.warn(`⚠️ Model ${model} not found. Available models: ${availableModels.join(', ')}`);
                console.log(`💡 Run: ollama pull ${model}`);
                return false;
            }

            console.log(`✅ Ollama is running with model: ${model}`);
            return true;

        } catch (error) {
            console.error(`❌ Ollama check failed: ${error.message}`);
            console.log(`💡 Make sure Ollama is running: ollama serve`);
            return false;
        }
    }

    /**
     * Google Gemini API integration
     */
    async callGeminiAPI(prompt) {
        const { GoogleGenerativeAI } = await import('@google/generative-ai');

        if (!process.env.GEMINI_API_KEY) {
            throw new Error('GEMINI_API_KEY environment variable is required');
        }

        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({
            model: this.options.model || 'gemini-1.5-pro',
            generationConfig: {
                temperature: 0.1, // Low temperature for consistent code analysis
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 4096,
            }
        });

        const result = await model.generateContent([
            {
                text: `${prompt}

IMPORTANT: You must respond with valid JSON only. No markdown, no explanations outside the JSON. The response should be a single JSON object with the exact structure requested.`
            }
        ]);

        const response = await result.response;
        const text = response.text();

        try {
            // Clean the response to extract JSON
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
                throw new Error('No JSON found in response');
            }

            return JSON.parse(jsonMatch[0]);
        } catch (parseError) {
            console.error('❌ Failed to parse Gemini response as JSON:', parseError.message);
            console.error('Raw response:', text);
            return this.getFallbackAnalysis();
        }
    }

    /**
     * OpenAI API integration (for comparison)
     */
    async callOpenAIAPI(prompt) {
        const { OpenAI } = await import('openai');

        if (!process.env.OPENAI_API_KEY) {
            throw new Error('OPENAI_API_KEY environment variable is required');
        }

        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });

        const response = await openai.chat.completions.create({
            model: this.options.model || 'gpt-4',
            messages: [
                {
                    role: 'system',
                    content: 'You are an expert code analyst. Always respond with valid JSON only.'
                },
                {
                    role: 'user',
                    content: prompt
                }
            ],
            temperature: 0.1,
            max_tokens: 4000
        });

        const content = response.choices[0].message.content;
        return JSON.parse(content);
    }

    /**
     * Anthropic Claude API integration
     */
    async callAnthropicAPI(prompt) {
        const { Anthropic } = await import('@anthropic-ai/sdk');

        if (!process.env.ANTHROPIC_API_KEY) {
            throw new Error('ANTHROPIC_API_KEY environment variable is required');
        }

        const anthropic = new Anthropic({
            apiKey: process.env.ANTHROPIC_API_KEY,
        });

        const response = await anthropic.messages.create({
            model: this.options.model || 'claude-3-sonnet-20240229',
            max_tokens: 4000,
            temperature: 0.1,
            messages: [
                {
                    role: 'user',
                    content: `${prompt}\n\nIMPORTANT: Respond with valid JSON only.`
                }
            ]
        });

        return JSON.parse(response.content[0].text);
    }

    /**
     * Fallback analysis when LLM calls fail
     */
    getFallbackAnalysis() {
        return {
            impactedFiles: ['Unable to determine - LLM analysis failed'],
            impactedFunctions: ['Please check manually'],
            riskLevel: 'medium',
            testingRecommendations: [
                'Run comprehensive test suite',
                'Manual verification recommended due to analysis failure'
            ],
            deploymentNotes: [
                'Extra caution recommended - automated analysis unavailable'
            ],
            confidence: 0.1,
            explanation: 'LLM analysis failed. This is a fallback response. Please review changes manually.'
        };
    }

    /**
     * Fallback to git status if no diff available
     */
    async getGitStatus() {
        return new Promise((resolve, reject) => {
            const gitProcess = spawn('git', ['status', '--porcelain'], {
                stdio: 'pipe',
                cwd: this.projectPath
            });

            let output = '';
            gitProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            gitProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(output);
                } else {
                    reject(new Error('No git changes detected'));
                }
            });
        });
    }
}
