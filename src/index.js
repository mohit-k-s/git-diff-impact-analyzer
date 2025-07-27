import { AIGitImpactAnalyzer } from './core/AIGitImpactAnalyzer.js';

/**
 * Demo: AI-powered Git Diff Impact Analysis with Ollama Local Models
 * Shows how to use local models for private, cost-free code analysis
 */

async function demonstrateAIImpactAnalysis() {
    console.log('🚀 AI-Powered Git Diff Impact Analysis Demo\n');
    const gitProjectFolder = '/code/repo'; // Just change this to your project full path
    const options = {
        llmProvider: 'gemini',
        model: 'gemini-2.5-flash-lite',
        ollamaUrl: 'http://localhost:11434',
        batchSize: 3,
        maxTokensPerBatch: 6000,
        rateLimitDelay: 500,
        maxRetries: 2,
        retryDelay: 1000
    }
    const aiAnalyzer = new AIGitImpactAnalyzer(gitProjectFolder, options);

    try {
        // Analyze current git diff with local AI
        console.log('\n🤖 Analyzing with ');
        const result = await aiAnalyzer.analyzeGitDiffWithAI();

        if (result.error) {
            console.log('❌ Analysis failed:', result.error);
            return;
        }

        if (result.message) {
            console.log('ℹ️', result.message);
            return;
        }

        // Display results
        console.log('\n📊 Impact Analysis Results:');
        console.log('='.repeat(50));
        console.log(`📁 Changed Files: ${result.changedFiles}`);
        console.log(`🎯 Confidence Level: ${(result.impactAnalysis.confidence * 100).toFixed(1)}%`);
        console.log(`⚠️  Risk Level: ${result.impactAnalysis.riskLevel.toUpperCase()}`);

        if (result.impactAnalysis.batchCount) {
            console.log(`📦 Processed Batches: ${result.impactAnalysis.successfulBatches}/${result.impactAnalysis.batchCount}`);
        }

        console.log('\n🔍 Impacted Files:');
        result.impactAnalysis.impactedFiles.forEach(file => {
            console.log(`  • ${file}`);
        });

        console.log('\n⚙️  Impacted Functions:');
        result.impactAnalysis.impactedFunctions.forEach(func => {
            console.log(`  • ${func}`);
        });

        console.log('\n🧪 Testing Recommendations:');
        result.impactAnalysis.testingRecommendations.forEach(test => {
            console.log(`  • ${test}`);
        });

        console.log('\n🚀 Deployment Notes:');
        result.impactAnalysis.deploymentNotes.forEach(note => {
            console.log(`  • ${note}`);
        });

        console.log('\n💡 AI Explanation:');
        console.log(result.impactAnalysis.explanation);

        console.log('\n✅ Analysis completed!');

    } catch (error) {
        console.error('❌ Run failed:', error.message);
    }
}


// Run the demo
async function main() {
    await demonstrateAIImpactAnalysis();
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
