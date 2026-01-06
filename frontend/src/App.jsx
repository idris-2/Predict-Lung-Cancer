import { useState } from 'react'
import './assets/App.css'
import Spinner from './components/Spinner'

function App() {
  const [formData, setFormData] = useState({
    gender: '',
    age: '',
    smoking: false,
    yellow_fingers: false,
    anxiety: false,
    peer_pressure: false,
    chronic_disease: false,
    fatigue: false,
    allergy: false,
    wheezing: false,
    alcohol: false,
    coughing: false,
    shortness_of_breath: false,
    swallowing_difficulty: false,
    chest_pain: false,
  })

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }))
    setError(null)
  }

const handleSubmit = async (e) => {
  e.preventDefault()
  setError(null)
  setLoading(true)

  // Validation
  if (!formData.gender || !formData.age) {
    setError('Please fill in all required fields')
    setLoading(false)
    return
  }

  if (formData.age < 1 || formData.age > 150) {
    setError('Please enter a valid age')
    setLoading(false)
    return
  }

  try {
    // Convert form data to the backend format
    const payload = {
      gender: formData.gender === 'Male' ? 1 : 0,
      age: parseInt(formData.age),
      smoking: formData.smoking ? 1 : 0,
      yellow_fingers: formData.yellow_fingers ? 1 : 0,
      anxiety: formData.anxiety ? 1 : 0,
      peer_pressure: formData.peer_pressure ? 1 : 0,
      chronic_disease: formData.chronic_disease ? 1 : 0,
      fatigue: formData.fatigue ? 1 : 0,
      allergy: formData.allergy ? 1 : 0,
      wheezing: formData.wheezing ? 1 : 0,
      alcohol: formData.alcohol ? 1 : 0,
      coughing: formData.coughing ? 1 : 0,
      shortness_of_breath: formData.shortness_of_breath ? 1 : 0,
      swallowing_difficulty: formData.swallowing_difficulty ? 1 : 0,
      chest_pain: formData.chest_pain ? 1 : 0,
    }

    const response = await fetch('https://nopen5446-lung-cancer.hf.space/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      throw new Error('Failed to get prediction')
    }

    const data = await response.json()
    setResult(data)
  } catch (err) {
    setError(err.message || 'An error occurred. Make sure your backend is running.')
  } finally {
    setLoading(false)
  }
}

  const handleReset = () => {
    setFormData({
      gender: '',
      age: '',
      smoking: false,
      yellow_fingers: false,
      anxiety: false,
      peer_pressure: false,
      chronic_disease: false,
      fatigue: false,
      allergy: false,
      wheezing: false,
      alcohol: false,
      coughing: false,
      shortness_of_breath: false,
      swallowing_difficulty: false,
      chest_pain: false,
    })
    setResult(null)
    setError(null)
  }

  const symptomFields = [
    { name: 'smoking', label: 'Smoking' },
    { name: 'yellow_fingers', label: 'Yellow Fingers' },
    { name: 'anxiety', label: 'Anxiety' },
    { name: 'peer_pressure', label: 'Peer Pressure' },
    { name: 'chronic_disease', label: 'Chronic Disease' },
    { name: 'fatigue', label: 'Fatigue' },
    { name: 'allergy', label: 'Allergy' },
    { name: 'wheezing', label: 'Wheezing' },
    { name: 'alcohol', label: 'Alcohol Use' },
    { name: 'coughing', label: 'Coughing' },
    { name: 'shortness_of_breath', label: 'Shortness of Breath' },
    { name: 'swallowing_difficulty', label: 'Swallowing Difficulty' },
    { name: 'chest_pain', label: 'Chest Pain' },
  ]

  return (
    <div className="container">
      <div className="form-wrapper">
        <div className="header">
          {result ? (
            <>
              <h1>Prediction Results</h1>
              <p className="subtitle">
                Based on your provided information and symptoms
              </p>
            </>
          ) : (
            <>
              <h1>Lung Cancer Prediction</h1>
              <p className="subtitle">
                Please enter your information and symptoms
              </p>
            </>
          )}
        </div>

      {result ? (
        <div className="result-container">
          <div className="prediction-layout">

            {/* Individual model scores */}
            <div className="model-scores">
              {Object.entries(result.individual_model_probs).map(
                ([model, value]) => (
                  <div key={model} className="model-card">
                    <h3>{model}</h3>
                    <p className="model-score">
                      {(value * 100).toFixed(1)}%
                    </p>
                  </div>
                )
              )}
            </div>

            {/* Overall score */}
            <div className={`overall-score ${result.risk_level}`}>
              <h2>Overall Risk</h2>
              <p className="overall-percentage">
                {(result.lung_cancer_probability * 100).toFixed(1)}%
              </p>
              <span className="risk-label">
                {result.risk_level.toUpperCase()}
              </span>
            </div>

            {/* Explanation */}
            <div className="prediction-text">
              {result.risk_level === 'high' && (
                <p>
                  The models indicate a <strong>high probability</strong> of lung cancer based on
                  the information you provided. This result does <strong>not</strong> mean that
                  you have lung cancer, but it suggests that your symptoms and risk factors are
                  commonly associated with higher-risk cases.
                  <br /><br />
                  This tool is intended as an <strong>early awareness and screening aid</strong>,
                  not a medical diagnosis. If you are experiencing symptoms or feel concerned
                  about your health, it is strongly recommended that you consult a qualified
                  medical professional for proper testing and evaluation.
                </p>
              )}

              {result.risk_level === 'medium' && (
                <p>
                  The prediction suggests a <strong>moderate risk</strong> of lung cancer based on
                  the patterns identified by the models. This means that some of your inputs
                  resemble cases where medical follow-up was beneficial.
                  <br /><br />
                  Please note that this system is <strong>not a doctor</strong> and cannot replace
                  professional medical advice. Consider monitoring your symptoms closely and
                  speaking with a healthcare provider if symptoms persist, worsen, or cause
                  concern.
                </p>
              )}

              {result.risk_level === 'low' && (
                <p>
                  The prediction indicates a <strong>low risk</strong> of lung cancer based on the
                  provided information. This suggests that your current inputs do not strongly
                  match high-risk patterns seen in the data.
                  <br /><br />
                  However, this result should not be interpreted as a guarantee of good health.
                  This model is designed for <strong>informational purposes only</strong>. If you
                  notice new or worsening symptoms, or have ongoing health concerns, seeking
                  professional medical advice is always recommended.
                </p>
              )}
            </div>

            <p style={{ fontSize: '0.85em', opacity: 0.7, marginTop: '16px', color: 'black', textAlign: 'center' }}>
              This application is intended for educational and informational purposes only and
              should not be used as a substitute for professional medical advice, diagnosis, or treatment.
            </p>

            {/* Action */}
            <button
              onClick={handleReset}
              className="btn btn-primary result-btn"
            >
              New Prediction
            </button>

          </div>
        </div>
      ) : (
          <form onSubmit={handleSubmit} className="form">
            {/* Personal Information Section */}
            <div className="form-section">
              <h2 className="section-title">Personal Information</h2>
              
              <div className="personal-info-grid">
                <div className="form-group">
                  <label htmlFor="gender">Gender <span className="required">*</span></label>
                  <select
                    id="gender"
                    name="gender"
                    value={formData.gender}
                    onChange={handleInputChange}
                    className="form-control"
                    required
                  >
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="age">Age <span className="required">*</span></label>
                  <input
                    type="number"
                    id="age"
                    name="age"
                    min="1"
                    max="150"
                    value={formData.age}
                    onChange={handleInputChange}
                    className="form-control"
                    placeholder="Enter your age"
                    required
                  />
                </div>
              </div>
            </div>

            {/* Symptoms Section */}
            <div className="form-section">
              <h2 className="section-title">Symptoms</h2>
              <p className="section-description">Check the symptoms you currently experience:</p>
              
              <div className="symptoms-grid">
                {symptomFields.map(field => (
                  <div key={field.name} className="checkbox-group">
                    <input
                      type="checkbox"
                      id={field.name}
                      name={field.name}
                      checked={formData[field.name]}
                      onChange={handleInputChange}
                      className="checkbox-input"
                    />
                    <label htmlFor={field.name} className="checkbox-label">
                      {field.label}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Form Actions */}
            <div className="form-actions">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? <Spinner size={20} /> : 'Get Prediction'}
              </button>
              <button
                type="reset"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Clear Form
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}

export default App
