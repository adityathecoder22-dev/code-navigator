const jwt = require('jsonwebtoken');

/**
 * Validates the Bearer JWT on incoming requests and attaches the user to req.user.
 * Returns 401 when the token is missing or invalid.
 */
function authenticate(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) {
    return res.status(401).json({ error: 'missing token' });
  }
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    return next();
  } catch (err) {
    return res.status(401).json({ error: 'invalid token' });
  }
}

module.exports = { authenticate };
