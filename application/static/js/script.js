// Fetch model accuracy from the API
async function fetchModelAccuracy() {
    try {
        const response = await fetch('/model-accuracy');
        const data = await response.json();
        if (data && data.accuracy) {
            modelAccuracy.textContent = `${data.accuracy}%`;
        }
    } catch (error) {
        console.error('Error fetching model accuracy:', error);
    }
}

// Make a prediction using the API
async function makePrediction(featuresData) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: featuresData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error making prediction:', error);
        throw error;
    }
}

// Display prediction result
function displayPredictionResult(data) {
    predictionResult.innerHTML = '';
    predictionResult.classList.remove('success', 'error');
    
    if (!data || !data.predictions || data.predictions.length === 0) {
        predictionResult.textContent = 'No prediction data received.';
        predictionResult.classList.add('error');
        return;
    }
    
    const prediction = data.predictions[0];
    const probabilities = data.probabilities || [];
    
    let resultHTML = `<h3>Prediction Result</h3>`;
    
    if (prediction === 1) {
        resultHTML += `<p><strong>Signal:</strong> <span class="bullish">BULLISH</span></p>`;
    } else if (prediction === 0) {
        resultHTML += `<p><strong>Signal:</strong> <span class="bearish">BEARISH</span></p>`;
    } else {
        resultHTML += `<p><strong>Signal:</strong> ${prediction}</p>`;
    }
    
    if (probabilities.length > 0) {
        resultHTML += `<p><strong>Confidence:</strong></p>`;
        resultHTML += `<ul>`;
        probabilities.forEach(prob => {
            for (const [label, value] of Object.entries(prob)) {
                const labelText = label === '1' ? 'Bullish' : 'Bearish';
                resultHTML += `<li>${labelText}: ${(value * 100).toFixed(2)}%</li>`;
            }
        });
        resultHTML += `</ul>`;
    }
    
    predictionResult.innerHTML = resultHTML;
    predictionResult.classList.add('success');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the dashboard
    initializeChart();
    fetchModelAccuracy();
    
    // Sample data selector
    sampleDataSelector.addEventListener('change', () => {
        const selectedOption = sampleDataSelector.value;
        if (selectedOption && sampleData[selectedOption]) {
            featuresInput.value = JSON.stringify(sampleData[selectedOption], null, 2);
        }
    });
    
    // Generate prediction button
    predictBtn.addEventListener('click', async () => {
        try {
            // Parse input JSON
            let featuresData;
            try {
                featuresData = JSON.parse(featuresInput.value);
            } catch (e) {
                predictionResult.textContent = 'Invalid JSON format. Please check your input.';
                predictionResult.classList.remove('success');
                predictionResult.classList.add('error');
                predictionResult.style.display = 'block';
                return;
            }
            
            // Make prediction
            const result = await makePrediction(featuresData);
            displayPredictionResult(result);
        } catch (error) {
            predictionResult.textContent = `Error: ${error.message}`;
            predictionResult.classList.remove('success');
            predictionResult.classList.add('error');
            predictionResult.style.display = 'block';
        }
    });
    
    // Refresh button
    refreshBtn.addEventListener('click', () => {
        fetchModelAccuracy();
        // Add animation to refresh button
        refreshBtn.classList.add('refreshing');
        setTimeout(() => {
            refreshBtn.classList.remove('refreshing');
        }, 1000);
    });
});
// Initialize chart
let predictionHistoryChart;

// Function to initialize the chart
function initializeChart() {
    const ctx = document.createElement('canvas');
    predictionChart.innerHTML = '';
    predictionChart.appendChild(ctx);
    
    predictionHistoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['1 Week Ago', '6 Days Ago', '5 Days Ago', '4 Days Ago', '3 Days Ago', '2 Days Ago', 'Yesterday', 'Today'],
            datasets: [{
                label: 'Trading Signal Predictions',
                data: [0.2, 0.4, 0.7, 0.6, 0.8, 0.3, 0.9, 0.7],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                tension: 0.3,
                pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Prediction Probability'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `Probability: ${(value * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}// DOM Elements
const refreshBtn = document.getElementById('refreshBtn');
const modelAccuracy = document.getElementById('modelAccuracy');
const bullishSignals = document.getElementById('bullishSignals');
const bearishSignals = document.getElementById('bearishSignals');
const sampleDataSelector = document.getElementById('sampleDataSelector');
const featuresInput = document.getElementById('featuresInput');
const predictBtn = document.getElementById('predictBtn');
const predictionResult = document.getElementById('predictionResult');
const predictionChart = document.getElementById('predictionChart');

// Sample Data Templates
const sampleData = {
    bullish: { "rsi_6_hl2": 78.0, "momentum_3": 0.8, "close_return_2": 0.7, "rsi_6_close": 75.0, "rsi_21_hl2": 73.0, "rsi_21_hlc3": 72.5, "rsi_14_hlc3": 70.8, "CCI_14_0.015": 135.0, "zscore_7": 1.2, "momentum_1": 0.5, "zscore_3": 1.3, "rsi_21_ohlc4": 74.0, "bbp": 0.9, "zscore_30": 1.1, "zscore_5": 1.2, "rsi_14_close": 73.0, "pvr": 1.3, "ADX_14": 28.0, "J": 85.0, "dec": -0.4, "rsi_14_hl2": 71.5, "inc": 0.6, "rsi_6_ohlc4": 76.0, "rolling_std_30": 1.0, "CMO_14": 60.0, "momentum_7": 0.9, "rsi_6_hlc3": 80.2, "zscore_2": 1.4, "zscore_9": 1.3, "zscore_14": 1.2, "rolling_std_9": 0.9, "rsi_14_ohlc4": 72.5, "DMN_14": 0.2, "BBP_20_2.0": 0.88, "zscore_21": 1.2, "willr": -15.0, "rsi_30_hl2": 74.0, "bop": 0.5, "close_return_1": 0.6, "close_return_3": 0.7, "DMP_14": 0.6, "rsi_30_ohlc4": 75.0, "momentum_2": 0.8, "cdl_doji": 1 },
    bearish: { "rsi_6_hl2": 22.5, "momentum_3": -0.6, "close_return_2": -0.5, "rsi_6_close": 25.0, "rsi_21_hl2": 28.0, "rsi_21_hlc3": 29.0, "rsi_14_hlc3": 28.4, "CCI_14_0.015": -125.0, "zscore_7": -1.3, "momentum_1": -0.4, "zscore_3": -1.2, "rsi_21_ohlc4": 27.0, "bbp": 0.22, "zscore_30": -1.1, "zscore_5": -1.0, "rsi_14_close": 28.5, "pvr": 0.6, "ADX_14": 14.0, "J": 10.0, "dec": 0.5, "rsi_14_hl2": 27.3, "inc": -0.5, "rsi_6_ohlc4": 26.0, "rolling_std_30": 3.5, "CMO_14": -55.0, "momentum_7": -0.7, "rsi_6_hlc3": 25.1, "zscore_2": -1.4, "zscore_9": -1.3, "zscore_14": -1.2, "rolling_std_9": 3.2, "rsi_14_ohlc4": 30.0, "DMN_14": 0.45, "BBP_20_2.0": 0.18, "zscore_21": -1.2, "willr": -85.0, "rsi_30_hl2": 28.0, "bop": -0.6, "close_return_1": -0.5, "close_return_3": -0.6, "DMP_14": 0.15, "rsi_30_ohlc4": 29.5, "momentum_2": -0.7, "cdl_doji": -1 },
    neutral: { "rsi_6_hl2": 50.0, "momentum_3": 0.0, "close_return_2": 0.0, "rsi_6_close": 50.0, "rsi_21_hl2": 50.0, "rsi_21_hlc3": 50.0, "rsi_14_hlc3": 50.3, "CCI_14_0.015": 3.0, "zscore_7": 0.0, "momentum_1": 0.0, "zscore_3": 0.0, "rsi_21_ohlc4": 50.0, "bbp": 0.49, "zscore_30": 0.0, "zscore_5": 0.0, "rsi_14_close": 50.0, "pvr": 1.0, "ADX_14": 20.0, "J": 50.0, "dec": 0.0, "rsi_14_hl2": 50.0, "inc": 0.0, "rsi_6_ohlc4": 50.0, "rolling_std_30": 2.0, "CMO_14": 0.0, "momentum_7": 0.0, "rsi_6_hlc3": 50.2, "zscore_2": 0.0, "zscore_9": 0.0, "zscore_14": 0.0, "rolling_std_9": 2.0, "rsi_14_ohlc4": 49.7, "DMN_14": 0.3, "BBP_20_2.0": 0.5, "zscore_21": 0.0, "willr": -50.0, "rsi_30_hl2": 50.0, "bop": 0.0, "close_return_1": 0.0, "close_return_3": 0.0, "DMP_14": 0.3, "rsi_30_ohlc4": 50.0, "momentum_2": 0.0, "cdl_doji": 0 }};