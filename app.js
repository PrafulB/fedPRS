import { Decentifai } from 'https://prafulb.github.io/decentifai/src/decentifai.js';

let decentifai = null;
let model = null;
let trainingData = null;
let testData = null;
let featureNames = [];
let snpColumns = [];
let convergenceChart = null;

window.decentifai = null;
window.model = null;

// Initialize Chart
const ctx = document.getElementById('convergenceChart').getContext('2d');
convergenceChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Loss',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                yAxisID: 'y',
            },
            {
                label: 'Accuracy',
                data: [],
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                yAxisID: 'y1',
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: { display: true, text: 'Loss', color: 'rgb(156, 163, 175)' },
                ticks: { color: 'rgb(156, 163, 175)' },
                grid: { color: 'rgba(75, 85, 99, 0.3)' }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: { display: true, text: 'Accuracy', color: 'rgb(156, 163, 175)' },
                ticks: { color: 'rgb(156, 163, 175)' },
                grid: { drawOnChartArea: false }
            },
            x: {
                ticks: { color: 'rgb(156, 163, 175)' },
                grid: { color: 'rgba(75, 85, 99, 0.3)' }
            }
        },
        plugins: {
            legend: {
                labels: { color: 'rgb(156, 163, 175)' }
            }
        }
    }
});

// File upload handler
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    logMessage(`Loading file: ${file.name}...`);
    document.getElementById('fileStatus').textContent = `Loading ${file.name}...`;

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
            processGWASData(results.data);
            document.getElementById('fileStatus').textContent = `✓ Loaded: ${file.name}`;
        },
        error: (error) => {
            logMessage(`Error loading file: ${error.message}`, 'error');
            document.getElementById('fileStatus').textContent = `✗ Error loading file`;
        }
    });
});

function processGWASData(data) {
    if (data.length === 0) {
        logMessage('Error: No data found in CSV', 'error');
        return;
    }

    const nonSnpColumns = new Set([
        'id', 
        'ageOfEntry', 
        'ageOfExit', 
        'gender', 
        'prs', 
        'case', 
        'ageOfOnset'
    ]);
    // Extract feature names
    const headers = Object.keys(data[0]);

    snpColumns = headers.filter(h => !nonSnpColumns.has(h));

    // Core features for GWAS
    const coreFeatures = ['ageOfEntry', 'ageOfExit', 'gender', 'PRS'];
    featureNames = [...coreFeatures, ...snpColumns];

    logMessage(`Found ${snpColumns.length} SNPs and ${featureNames.length} total features`);

    // Split data into train/test (80/20)
    const shuffled = data.sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * 0.8);

    const trainRaw = shuffled.slice(0, splitIndex);
    const testRaw = shuffled.slice(splitIndex);

    // Prepare training data
    trainingData = {
        features: trainRaw.map(row => featureNames.map(f => row[f] || 0)),
        labels: trainRaw.map(row => row.case || 0)
    };

    // Prepare test data
    testData = {
        features: testRaw.map(row => featureNames.map(f => row[f] || 0)),
        labels: testRaw.map(row => row.case || 0)
    };

    document.getElementById('sampleCount').textContent = data.length;
    logMessage(`Data processed: ${trainRaw.length} training, ${testRaw.length} test samples`);

    // Enable connect button
    document.getElementById('connectBtn').disabled = false;
}

function createLogisticRegressionModel(numSNPs) {
    // ... (This function is identical to the one in the previous response)
    console.log("Creating TensorFlow.js logistic regression model...");
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [numSNPs],
        units: 1,
        activation: 'sigmoid'
    }));
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy', 'loss']
    });
    model.train = async (trainingArgs) => {
        const { data: { xs, ys }, options } = trainingArgs;
        return await model.fit(xs, ys, options);
    };
    return model;
}

window.initializeFederation = async function () {
    if (!trainingData) {
        logMessage('Please load GWAS data first', 'error');
        return;
    }

    const roomId = document.getElementById('roomId').value;
    const selfName = document.getElementById('selfName').value;
    const minPeers = parseInt(document.getElementById('minPeers').value);
    const maxRounds = parseInt(document.getElementById('maxRounds').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);

    logMessage('Creating logistic regression model...');

    // Create TensorFlow.js model
    model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [featureNames.length],
                units: 1,
                activation: 'sigmoid',
                kernelInitializer: 'glorotUniform'
            })
        ]
    });

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Add methods for Decentifai
    model.getLoss = async function () {
        if (!testData) return null;
        const xs = tf.tensor2d(testData.features);
        const ys = tf.tensor2d(testData.labels, [testData.labels.length, 1]);
        const result = model.evaluate(xs, ys);
        const loss = await result[0].data();
        xs.dispose();
        ys.dispose();
        result[0].dispose();
        result[1].dispose();
        return loss[0];
    };

    model.getAccuracy = async function () {
        if (!testData) return null;
        const xs = tf.tensor2d(testData.features);
        const ys = tf.tensor2d(testData.labels, [testData.labels.length, 1]);
        const result = model.evaluate(xs, ys);
        const accuracy = await result[1].data();
        xs.dispose();
        ys.dispose();
        result[0].dispose();
        result[1].dispose();
        return accuracy[0];
    };
    // model.train = model.fit 

    logMessage('Initializing Decentifai federation...');

    try {
        model.train = async (trainingArgs) => {
            const { data: { features, labels }, options } = trainingArgs;
            return await model.fit(tf.tensor2d(features), tf.tensor2d(labels, [trainingData.labels.length, 1]), options);
        };

        decentifai = new Decentifai({
            roomId: roomId,
            backend: 'tfjs',
            metadata: {
                name: selfName
            },
            model: model,
            trainingData: trainingData,
            testData: testData,
            trainingOptions: {
                epochs: 5,
                batchSize: 5000,
                verbose: 0,
                shuffle: true
            },
            federationOptions: {
                minPeers: minPeers,
                maxPeers: 10,
                minRounds: 5,
                maxRounds: maxRounds,
                waitTime: 2000,
                convergenceThresholds: {
                    lossDelta: 0.01,
                    accuracyDelta: 0.001,
                    stabilityWindow: 3
                }
            },
            autoTrain: false,
            debug: true
        });

        window.decentifai = decentifai;
        window.model = model;
        if (!selfName) {
            document.getElementById("selfName").value = decentifai.ydoc.clientID
        }
        setupEventListeners();

        document.getElementById('status').textContent = 'Connected';
        document.getElementById('status').className = 'text-lg font-semibold text-green-400';
        document.getElementById('connectBtn').disabled = true;
        document.getElementById('startBtn').disabled = false;
        document.getElementById('evaluateBtn').disabled = false;

        logMessage('✓ Successfully connected to federation');

    } catch (error) {
        logMessage(`Error initializing federation: ${error.message}`, 'error');
    }
};

function setupEventListeners() {
    decentifai.on('peersAdded', (e) => {
        logMessage(`✓ Peer connected: ${e.detail.name || e.detail.peerId}`);
        updatePeersList();
    });

    decentifai.on('peersRemoved', (e) => {
        logMessage(`✗ Peer disconnected: ${e.detail.name || e.detail.peerId}`);
        updatePeersList();
    });

    decentifai.on('roundStarted', (e) => {
        logMessage(`→ Round ${e.detail.round} started`, 'info');
    });

    decentifai.on('localTrainingCompleted', async (e) => {
        logMessage(`✓ Local training completed for round ${e.detail.round}`);
        // await updateMetrics();
    });

    decentifai.on('roundFinalized', async (e) => {
        logMessage(`✓ Round ${e.detail.round} finalized with ${e.detail.participants} peers`);
        document.getElementById('currentRound').textContent = e.detail.round;
        await updateMetrics();
    });

    decentifai.on('modelConverged', (e) => {
        logMessage(`🎉 Model converged at round ${e.detail.round}!`, 'success');
        document.getElementById('status').textContent = 'Converged';
        document.getElementById('status').className = 'text-lg font-semibold text-yellow-400';
        displaySNPEffects();
    });

    decentifai.on('autoTrainingStopped', (e) => {
        logMessage(`Training stopped: ${e.detail.reason}`, 'info');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        displaySNPEffects();
    });

    decentifai.on('autoTrainingError', (e) => {
        logMessage(`Training error: ${e.detail.error}`, 'error');
    });
}

function updatePeersList() {
    const peers = decentifai.getPeers();
    const peerCount = Object.keys(peers).length;
    document.getElementById('peerCount').textContent = peerCount;

    const peersList = document.getElementById('peersList');
    if (peerCount === 0) {
        peersList.innerHTML = '<p class="text-gray-500">No peers connected</p>';
    } else {
        peersList.innerHTML = Object.values(peers).map(peer => `
                    <div class="bg-gray-900 rounded px-3 py-2">
                        <span class="text-green-400">●</span> 
                        ${peer.metadata?.name || 'Peer-' + peer.clientID}
                    </div>
                `).join('');
    }
}

async function updateMetrics() {
    if (!model || !testData) return;

    const loss = await model.getLoss();
    const accuracy = await model.getAccuracy();

    document.getElementById('currentLoss').textContent = loss.toFixed(4);
    document.getElementById('currentAccuracy').textContent = (accuracy * 100).toFixed(2) + '%';

    // Update chart
    const round = decentifai.getCurrentRound();
    convergenceChart.data.labels.push(`R${round}`);
    convergenceChart.data.datasets[0].data.push(loss);
    convergenceChart.data.datasets[1].data.push(accuracy);

    if (convergenceChart.data.labels.length > 20) {
        convergenceChart.data.labels.shift();
        convergenceChart.data.datasets[0].data.shift();
        convergenceChart.data.datasets[1].data.shift();
    }

    convergenceChart.update();
}

window.startTraining = function () {
    if (!decentifai) return;

    logMessage('Starting auto-training...', 'info');
    decentifai.setAutoTraining(true);

    document.getElementById('status').textContent = 'Training';
    document.getElementById('status').className = 'text-lg font-semibold text-purple-400';
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
};

window.stopTraining = function () {
    if (!decentifai) return;

    logMessage('Stopping auto-training...', 'info');
    decentifai.setAutoTraining(false);

    document.getElementById('status').textContent = 'Connected';
    document.getElementById('status').className = 'text-lg font-semibold text-green-400';
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
};

window.evaluateModel = async function () {
    if (!model || !testData) return;

    logMessage('Evaluating model on test set...', 'info');

    const xs = tf.tensor2d(testData.features);
    const ys = tf.tensor2d(testData.labels, [testData.labels.length, 1]);

    const predictions = model.predict(xs);
    const predArray = await predictions.data();

    // Calculate metrics
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < testData.labels.length; i++) {
        const pred = predArray[i] > 0.5 ? 1 : 0;
        const actual = testData.labels[i];

        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    }

    const accuracy = (tp + tn) / testData.labels.length;
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;

    logMessage(`Evaluation Results:`, 'success');
    logMessage(`  Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    logMessage(`  Precision: ${(precision * 100).toFixed(2)}%`);
    logMessage(`  Recall: ${(recall * 100).toFixed(2)}%`);
    logMessage(`  F1 Score: ${f1.toFixed(4)}`);

    xs.dispose();
    ys.dispose();
    predictions.dispose();

    displaySNPEffects();
};

function displaySNPEffects() {
    if (!model) return;

    const weights = model.getWeights()[0];
    const weightsArray = weights.arraySync();

    const snpEffects = featureNames.map((name, idx) => ({
        name: name,
        coefficient: weightsArray[idx][0],
        absCoefficient: Math.abs(weightsArray[idx][0])
    }))
        .filter(effect => snpColumns.includes(effect.name))
        .sort((a, b) => b.absCoefficient - a.absCoefficient)
        .slice(0, 20);

    const snpEffectsDiv = document.getElementById('snpEffects');
    snpEffectsDiv.innerHTML = snpEffects.map((effect, idx) => `
                <div class="flex justify-between items-center py-2 border-b border-gray-800">
                    <span class="text-sm font-mono">${idx + 1}. ${effect.name}</span>
                    <span class="text-sm font-semibold ${effect.coefficient > 0 ? 'text-red-400' : 'text-blue-400'}">
                        ${effect.coefficient > 0 ? '+' : ''}${effect.coefficient.toFixed(6)}
                    </span>
                </div>
            `).join('');
}

function logMessage(message, type = 'log') {
    const log = document.getElementById('trainingLog');
    const timestamp = new Date().toLocaleTimeString();
    const colors = {
        log: 'text-gray-300',
        info: 'text-blue-400',
        success: 'text-green-400',
        error: 'text-red-400'
    };

    const entry = document.createElement('p');
    entry.className = colors[type] || colors.log;
    entry.textContent = `[${timestamp}] ${message}`;

    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

// Initial log message
logMessage('Application loaded. Upload GWAS data to begin.', 'info');