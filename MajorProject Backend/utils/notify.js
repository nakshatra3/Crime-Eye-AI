// utils/notify.js
const twilio = require('twilio');
const nodemailer = require('nodemailer');

const client = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH_TOKEN);
const fallbackPhone = '+917745064813';

exports.sendSMS = (to, type, location, message) => {
  return client.messages.create({
    body: `🚨 Alert: ${type}\n📍 Location: ${location.name || `${location.latitude}, ${location.longitude}`}\n📝 Message: ${message}`,
    from: process.env.TWILIO_PHONE,
    to: to || fallbackPhone,
  });
};

exports.sendEmail = async (to, type, location, message, image_url = null) => {
  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS
    }
  });

  let htmlContent = `
    <h2>🚨 Alert: ${type}</h2>
    <p><strong>Location:</strong> ${location.name || `${location.latitude}, ${location.longitude}`}</p>
    <p><strong>Message:</strong> ${message}</p>
  `;

  if (image_url) {
    htmlContent += `<img src="${image_url}" alt="Incident Image" width="300"/>`;
  }

  await transporter.sendMail({
    from: process.env.EMAIL_USER,
    to: to || process.env.EMAIL_USER,
    subject: `🚨 Alert: ${type}`,
    html: htmlContent,
  });
};
