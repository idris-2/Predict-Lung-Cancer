import { useState } from 'react'
import './App.css'

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
      // Convert checkbox values to 1 or 0 for the backend
      const payload = {
        gender: formData.gender,
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

      // Send to backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
          <h1>Lung Cancer Prediction</h1>
          <p className="subtitle">Please enter your information and symptoms</p>
        </div>

        {result ? (
          <div className="result-container">
            <div className="result-card">
              <h2>Prediction Result</h2>
              <div className="result-content">
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </div>
              <button onClick={handleReset} className="btn btn-primary">
                New Prediction
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="form">
            {/* Personal Information Section */}
            <div className="form-section">
              <h2 className="section-title">Personal Information</h2>
              
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
                {loading ? 'Sending...' : 'Get Prediction'}
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
