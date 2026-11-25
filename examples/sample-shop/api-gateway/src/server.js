const express = require('express');
const { authenticate } = require('./middleware/auth');
const ordersRouter = require('./routes/orders');
const productsRouter = require('./routes/products');

const app = express();
app.use(express.json());

// Health check used by Kubernetes probes
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.use('/api/orders', authenticate, ordersRouter);
app.use('/api/products', productsRouter);

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`api-gateway listening on ${PORT}`));
