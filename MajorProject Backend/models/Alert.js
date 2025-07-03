// models/Alert.js
const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  type: String,
  location: {
    latitude: Number,
    longitude: Number
  },
  timestamp: Date,
  image_url: String,
  notified: Boolean
});

module.exports = mongoose.model('Alert', alertSchema);
