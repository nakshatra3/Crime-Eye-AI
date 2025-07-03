require('dotenv').config();
const express = require('express');
const connectDB = require('./config/db');
const alertRoutes = require('./routes/alertRoutes');

const app = express();
const PORT = process.env.PORT || 5000;

// DB Connect
connectDB();

// Middleware
app.use(express.json());

// Routes
app.use('/api/alerts', alertRoutes);

// Server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
