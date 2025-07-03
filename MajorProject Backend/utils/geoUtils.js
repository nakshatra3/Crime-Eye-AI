const axios = require('axios');

exports.getNearestEmergencyService = async (lat, lng) => {
  const apiKey = process.env.GOOGLE_MAPS_API_KEY;

  const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${lat},${lng}&radius=10000&type=police&key=${apiKey}`;

  try {
    const { data } = await axios.get(url);

    if (!data || !data.results || data.results.length === 0) {
      console.warn("⚠️ No emergency service found near:", lat, lng);
      return "Nearest Police Station (Not Found)";
    }

    return data.results[0].name;
  } catch (err) {
    console.error("❌ Error fetching from Google Maps API:", err.message);
    return "Emergency Service (Unknown)";
  }
};
