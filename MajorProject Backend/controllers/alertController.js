// controllers/alertController.js
const Alert = require('../models/Alert');
const { getNearestEmergencyService } = require('../utils/geoUtils');
const { sendSMS, sendEmail } = require('../utils/notify');

exports.createAlert = async (req, res) => {
  try {
    const { type, location, timestamp, image_url, message } = req.body;

    const nearest = await getNearestEmergencyService(location, type);

    // Updated calls with message
    await sendSMS(nearest.phone, type, location, message);
    await sendEmail(nearest.email, type, location, message, image_url);

    const alert = new Alert({
      type,
      location,
      timestamp,
      image_url,
      message,
      notified: true
    });

    await alert.save();

    res.status(201).json({ message: 'Alert created and notification sent.' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create alert.' });
  }
};
