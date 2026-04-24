// Storage key for localStorage
const STORAGE_KEY = 'insight_absa_reviews';

// Load history from localStorage on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});

// Analyze review
async function analyze() {
    const textarea = document.getElementById('inputText');
    const text = textarea.value.trim();
    const resultsDiv = document.getElementById('results');
    
    if (!text) {
        resultsDiv.innerHTML = '<div class="empty-state">✏️ Please enter a review to analyze</div>';
        return;
    }
    
    resultsDiv.innerHTML = '<div class="loading">🔍 Analyzing sentiment</div>';
    
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error('Server error');
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
        // Save to history
        saveToHistory(data);
        
        // Clear input
        textarea.value = '';
        
        // Reload history display
        loadHistory();
        
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = '<div class="empty-state" style="color: #f44336;">❌ Error connecting to server. Make sure the backend is running on port 5000</div>';
    }
}

// Display analysis results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    const aspectIcons = {
        food: '🍔',
        service: '🧑‍🍳',
        price: '💰',
        ambiance: '🎶',
        cleanliness: '🧼',
        delivery: '🚚',
        experience: '⭐'
    };
    
    let aspectsHtml = '';
    
    for (const [aspect, sentiment] of Object.entries(data.aspect_sentiments)) {
        if (sentiment !== 'Not Mentioned') {
            aspectsHtml += `
                <div class="aspect-item">
                    <span class="aspect-name">
                        ${aspectIcons[aspect] || '📌'} ${aspect.charAt(0).toUpperCase() + aspect.slice(1)}
                    </span>
                    <span class="aspect-sentiment ${sentiment}">${sentiment}</span>
                </div>
            `;
        }
    }
    
    const html = `
        <div class="result-card">
            <div class="overall-sentiment ${data.overall_sentiment}">
                📊 Overall Sentiment: ${data.overall_sentiment}
            </div>
            <div style="margin-top: 1rem;">
                <strong style="color: #6c63ff;">Aspect Analysis:</strong>
                <div class="aspects-grid">
                    ${aspectsHtml}
                </div>
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
}

// Save review to localStorage
function saveToHistory(data) {
    let history = JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    
    const reviewData = {
        id: Date.now(),
        text: data.original_text,
        overall_sentiment: data.overall_sentiment,
        aspects: data.aspects,
        aspect_sentiments: data.aspect_sentiments,
        timestamp: new Date().toLocaleString()
    };
    
    history.unshift(reviewData); // Add to beginning
    
    // Keep only last 20 reviews
    if (history.length > 20) {
        history = history.slice(0, 20);
    }
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

// Load and display history
function loadHistory() {
    const historyDiv = document.getElementById('history');
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    
    if (history.length === 0) {
        historyDiv.innerHTML = '<div class="empty-state">📭 No reviews analyzed yet. Write your first review above!</div>';
        return;
    }
    
    const aspectIcons = {
        food: '🍔',
        service: '🧑‍🍳',
        price: '💰',
        ambiance: '🎶',
        cleanliness: '🧼',
        delivery: '🚚',
        experience: '⭐'
    };
    
    let html = '';
    
    history.forEach(review => {
        // Create aspects badges for this review
        let aspectsBadges = '';
        for (const [aspect, sentiment] of Object.entries(review.aspect_sentiments)) {
            if (sentiment !== 'Not Mentioned') {
                const sentimentClass = sentiment === 'Positive' ? 'Positive' : 
                                      sentiment === 'Negative' ? 'Negative' : 'Neutral';
                aspectsBadges += `
                    <span class="history-aspect ${sentimentClass}">
                        ${aspectIcons[aspect] || '📌'} ${aspect}: ${sentiment}
                    </span>
                `;
            }
        }
        
        // Truncate long text
        const displayText = review.text.length > 150 ? 
            review.text.substring(0, 150) + '...' : 
            review.text;
        
        html += `
            <div class="history-card" onclick="viewReviewDetails(${review.id})">
                <div class="history-review-text">
                    "${displayText}"
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <div class="history-details">
                        ${aspectsBadges}
                    </div>
                    <div style="font-size: 0.7rem; color: #6a6a8a;">
                        ${review.timestamp}
                    </div>
                </div>
            </div>
        `;
    });
    
    historyDiv.innerHTML = html;
}

// View full review details (modal style)
function viewReviewDetails(id) {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    const review = history.find(r => r.id === id);
    
    if (!review) return;
    
    const aspectIcons = {
        food: '🍔',
        service: '🧑‍🍳',
        price: '💰',
        ambiance: '🎶',
        cleanliness: '🧼',
        delivery: '🚚',
        experience: '⭐'
    };
    
    let aspectsHtml = '';
    for (const [aspect, sentiment] of Object.entries(review.aspect_sentiments)) {
        if (sentiment !== 'Not Mentioned') {
            aspectsHtml += `
                <div class="aspect-item">
                    <span class="aspect-name">
                        ${aspectIcons[aspect] || '📌'} ${aspect.charAt(0).toUpperCase() + aspect.slice(1)}
                    </span>
                    <span class="aspect-sentiment ${sentiment}">${sentiment}</span>
                </div>
            `;
        }
    }
    
    // Create modal dialog
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        animation: fadeIn 0.3s ease-out;
    `;
    
    modal.innerHTML = `
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #12121f 100%);
            border: 1px solid #6c63ff;
            border-radius: 20px;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80%;
            overflow-y: auto;
            position: relative;
        ">
            <button onclick="this.parentElement.parentElement.remove()" style="
                position: absolute;
                top: 1rem;
                right: 1rem;
                background: #2a2a3e;
                border: none;
                color: white;
                font-size: 1.5rem;
                cursor: pointer;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            ">×</button>
            
            <div class="overall-sentiment ${review.overall_sentiment}" style="margin-bottom: 1rem;">
                📊 Overall Sentiment: ${review.overall_sentiment}
            </div>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #6c63ff;">Review:</strong>
                <p style="margin-top: 0.5rem; line-height: 1.6;">${review.text}</p>
            </div>
            
            <div>
                <strong style="color: #6c63ff;">Detailed Analysis:</strong>
                <div class="aspects-grid" style="margin-top: 0.5rem;">
                    ${aspectsHtml}
                </div>
            </div>
            
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #6a6a8a;">
                Analyzed on: ${review.timestamp}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Clear all history (optional utility function)
function clearHistory() {
    if (confirm('Are you sure you want to clear all review history?')) {
        localStorage.removeItem(STORAGE_KEY);
        loadHistory();
        document.getElementById('results').innerHTML = '';
    }
}