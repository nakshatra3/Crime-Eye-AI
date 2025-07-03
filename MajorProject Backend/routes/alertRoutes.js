const express = require('express');
const router = express.Router();
const { createAlert } = require('../controllers/alertController');

// Define the route for POST /alert
router.post('/', createAlert);

module.exports = router;
