// import { Decentifai } from 'http://localhost:5505/src/decentifai.js';
import { Decentifai } from 'https://prafulb.github.io/decentifai/src/decentifai.js';

let decentifai = null;
let decentifaiAppId = "FedPRS"
let model = null;
let trainingData = null;
let testData = null;
let featureNames = [];
let snpColumns = [];
let convergenceChart = null;

window.decentifai = null;
window.model = null;

// --- Helper: AUC Calculation ---
function calculateAUC(yTrue, yPred) {
    // Combine arrays
    const data = yTrue.map((label, i) => ({ label, pred: yPred[i] }));
    // Sort by prediction score descending
    data.sort((a, b) => b.pred - a.pred);

    let posCount = 0;
    let negCount = 0;

    // Count positives and negatives
    data.forEach(item => {
        if (item.label === 1) posCount++;
        else negCount++;
    });

    let auc = 0;
    let posFound = 0;

    // Calculate Area via Mann-Whitney U equivalent / integration
    data.forEach(item => {
        if (item.label === 1) {
            posFound++;
        } else {
            // For every negative sample, we add the number of positives ranked higher than it
            auc += posFound;
        }
    });

    return (posCount === 0 || negCount === 0) ? 0 : auc / (posCount * negCount);
}

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
                borderColor: 'rgb(59, 130, 246)', // Blue
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                yAxisID: 'y',
                tension: 0.1
            },
            {
                label: 'AUC', // Changed from Accuracy
                data: [],
                borderColor: 'rgb(168, 85, 247)', // Purple
                backgroundColor: 'rgba(168, 85, 247, 0.1)',
                yAxisID: 'y1',
                tension: 0.1
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
                title: { display: true, text: 'AUC', color: 'rgb(156, 163, 175)' },
                ticks: { color: 'rgb(156, 163, 175)' },
                grid: { drawOnChartArea: false },
                min: 0,
                max: 1
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

    // NEW: Helper method to get AUC for the federation loop
    model.getAUC = async function () {
        if (!testData) return null;
        const xs = tf.tensor2d(testData.features);
        const predsTensor = model.predict(xs);
        const preds = await predsTensor.data();
        
        const auc = calculateAUC(testData.labels, preds);
        
        xs.dispose();
        predsTensor.dispose();
        return auc;
    };

    logMessage('Initializing Decentifai federation...');

    try {
        model.train = async (trainingArgs) => {
            const { data: { features, labels }, options } = trainingArgs;
            return await model.fit(tf.tensor2d(features), tf.tensor2d(labels, [trainingData.labels.length, 1]), options);
        };

        decentifai = new Decentifai({
            appId: decentifaiAppId,
            roomId: roomId,
            connectionType: 'webrtc',
            backend: 'tfjs',
            metadata: {
                name: selfName
            },
            model: model,
            trainingData: trainingData,
            testData: testData,
            trainingOptions: {
                epochs: 1,
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

    // Get Loss and AUC
    const loss = await model.getLoss();
    const auc = await model.getAUC();

    document.getElementById('currentLoss').textContent = loss.toFixed(4);
    document.getElementById('currentAUC').textContent = auc.toFixed(4);

    // Update chart
    const round = decentifai.getCurrentRound();
    convergenceChart.data.labels.push(`R${round}`);
    convergenceChart.data.datasets[0].data.push(loss);
    convergenceChart.data.datasets[1].data.push(auc);

    // Keep chart window sliding (last 20 points)
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
    // const ys = tf.tensor2d(testData.labels, [testData.labels.length, 1]); // Not used for predict

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
    const auc = calculateAUC(testData.labels, predArray);

    logMessage(`Evaluation Results:`, 'success');
    logMessage(`  Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    logMessage(`  Precision: ${(precision * 100).toFixed(2)}%`);
    logMessage(`  Recall: ${(recall * 100).toFixed(2)}%`);
    logMessage(`  F1 Score: ${f1.toFixed(4)}`);
    logMessage(`  AUC: ${auc.toFixed(4)}`); // Added AUC log

    xs.dispose();
    // ys.dispose();
    predictions.dispose();

    displaySNPEffects();
};

async function displaySNPEffects() {
    if (!model || !trainingData) return;

    const weights = model.getWeights()[0];
    const weightsArray = weights.arraySync();

    // 1. Filter to get Top 20 SNPs by absolute coefficient
    let topSnps = featureNames.map((name, idx) => ({
        name: name,
        coefficient: weightsArray[idx][0],
        absCoefficient: Math.abs(weightsArray[idx][0]),
        index: idx
    }))
    .filter(effect => snpColumns.includes(effect.name))
    .sort((a, b) => b.absCoefficient - a.absCoefficient)
    .slice(0, 20);

    // 2. Approximation of Standard Error (SE) for Confidence Intervals
    // We calculate SE ~ 1 / sqrt(Sum(x^2 * p * (1-p))) (Diagonal of Fisher Info Matrix)
    // Note: We use the training set for this calculation.
    // To keep UI responsive, we take a subset of up to 1000 samples.
    const sampleSize = Math.min(trainingData.features.length, 1000);
    const subsetFeatures = trainingData.features.slice(0, sampleSize);
    
    const xs = tf.tensor2d(subsetFeatures);
    const probsTensor = model.predict(xs);
    const probs = await probsTensor.data();
    xs.dispose();
    probsTensor.dispose();

    topSnps = topSnps.map(snp => {
        let hessianDiag = 0;
        const col = snp.index;

        for(let i = 0; i < sampleSize; i++) {
            const p = probs[i];
            const x = subsetFeatures[i][col];
            // Diagonal element of Hessian: sum(x_ij^2 * p_i * (1-p_i))
            hessianDiag += (x * x) * p * (1 - p);
        }

        // Prevent division by zero
        hessianDiag = Math.max(hessianDiag, 1e-6);
        
        const se = 1 / Math.sqrt(hessianDiag);
        // 95% Confidence Interval: Beta +/- 1.96 * SE
        return {
            ...snp,
            se: se,
            ciLower: snp.coefficient - (1.96 * se),
            ciUpper: snp.coefficient + (1.96 * se)
        };
    });

    // 3. Determine Plot Scales
    // We need the global min and max across all CIs to scale the forest plot
    const minVal = Math.min(0, ...topSnps.map(s => s.ciLower));
    const maxVal = Math.max(0, ...topSnps.map(s => s.ciUpper));
    const range = maxVal - minVal + 0.0001; // avoid div/0

    // 4. Render Forest Plot
    const snpEffectsDiv = document.getElementById('snpEffects');
    snpEffectsDiv.innerHTML = topSnps.map((effect, idx) => {
        // Calculate percentages for CSS positioning
        const leftPct = ((effect.ciLower - minVal) / range) * 100;
        const widthPct = ((effect.ciUpper - effect.ciLower) / range) * 100;
        const meanPct = ((effect.coefficient - minVal) / range) * 100;
        const zeroPct = ((0 - minVal) / range) * 100;

        return `
            <div class="py-3 border-b border-gray-800 grid grid-cols-12 gap-4 items-center">
                <!-- Name -->
                <div class="col-span-3 text-xs font-mono truncate" title="${effect.name}">
                    ${idx + 1}. ${effect.name}
                </div>
                
                <!-- Forest Plot Visualization -->
                <div class="col-span-6 relative h-6 bg-gray-800 rounded">
                    <!-- Zero Line -->
                    <div class="absolute top-0 bottom-0 border-l border-dashed border-gray-500" 
                         style="left: ${zeroPct}%; opacity: 0.5;"></div>
                    
                    <!-- CI Bar -->
                    <div class="absolute top-1/2 h-1 bg-blue-500 opacity-50 -translate-y-1/2 rounded"
                         style="left: ${leftPct}%; width: ${widthPct}%;"></div>
                         
                    <!-- Mean Dot -->
                    <div class="absolute top-1/2 w-2 h-2 rounded-full -translate-y-1/2 -translate-x-1/2 shadow-sm
                        ${effect.coefficient > 0 ? 'bg-red-400' : 'bg-blue-400'}"
                         style="left: ${meanPct}%;"></div>
                </div>

                <!-- Numeric Stats -->
                <div class="col-span-3 text-right text-xs">
                    <div class="font-bold ${effect.coefficient > 0 ? 'text-red-400' : 'text-blue-400'}">
                        ${effect.coefficient.toFixed(4)}
                    </div>
                    <div class="text-gray-600 scale-90 origin-right">
                        [${effect.ciLower.toFixed(2)}, ${effect.ciUpper.toFixed(2)}]
                    </div>
                </div>
            </div>
        `;
    }).join('');
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