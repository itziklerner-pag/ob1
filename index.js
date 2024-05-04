const WebSocket = require('ws');
const axios = require('axios');

//const depthSnapshotUrl = 'https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000';
const depthSnapshotUrl = ' https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000';

let orderBook = {
    bids: {},
    asks: {}
};

async function loadOrderBookSnapshot() {
    try {
        const response = await axios.get(depthSnapshotUrl);
        const { bids, asks } = response.data;

        bids.forEach(([price, quantity]) => {
            orderBook.bids[price] = quantity;
        });
        asks.forEach(([price, quantity]) => {
            orderBook.asks[price] = quantity;
        });

        console.log('Initial order book snapshot loaded');
    } catch (error) {
        console.error('Failed to load order book snapshot:', error);
    }
}

function logOrderBook() {
    const bids = Object.entries(orderBook.bids)
        .sort((a, b) => parseFloat(b[0]) - parseFloat(a[0])) // Sort by price descending
        .slice(0, 1) // Get top 20 bids
        .map(([price, quantity]) => [price, quantity.toString()]); // Format as strings in an array

    const asks = Object.entries(orderBook.asks)
        .sort((a, b) => parseFloat(a[0]) - parseFloat(b[0])) // Sort by price ascending
        .slice(0, 1) // Get bottom 20 asks
        .map(([price, quantity]) => [price, quantity.toString()]); // Format as strings in an array

    console.log(JSON.stringify({
        /*stream: "btcusdt@depth20@1000ms",
        data: {
            e: "depthUpdate",
            E: Date.now(), // Current timestamp as event time
           // T: Date.now(), // Current timestamp as transaction time
            s: "BTCUSDT",
            U: 0, // Placeholder for First update ID in event
            u: 0, // Placeholder for Last update ID in event
            pu: 0, // Placeholder for Last update ID in previous event */
            T: Date.now(),
            b: bids,
            a: asks
        //}
    }));
}

function processDepthUpdate(data) {
    const { b: bids, a: asks } = data;

    bids.forEach(([price, quantity]) => {
        if (parseFloat(quantity) === 0) {
            delete orderBook.bids[price];
        } else {
            orderBook.bids[price] = quantity;
        }
    });

    asks.forEach(([price, quantity]) => {
        if (parseFloat(quantity) === 0) {
            delete orderBook.asks[price];
        } else {
            orderBook.asks[price] = quantity;
        }
    });

   // console.log('Order Book updated');
}

let ws;
function initializeWebSocket() {
    //ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth');
    ws = new WebSocket('wss://fstream.binance.com/stream?streams=btcusdt@depth/btcusdt@aggTrade');

    ws.on('message', (message) => {
        const data = JSON.parse(message);
        if (data.e === "depthUpdate") {
            processDepthUpdate(data);
        } else if (data.e === "aggTrade") {
            processTradeUpdate(data);
        }
    });

    ws.on('ping', () => {
        ws.pong();  // Respond to pings with pongs
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });

    ws.on('open', () => {
        console.log('WebSocket connected');
    });

    ws.on('close', () => {
        console.log('WebSocket disconnected. Reconnecting...');
        setTimeout(initializeWebSocket, 1000);  // Attempt to reconnect after 1 second
    });
}

async function start() {
    await loadOrderBookSnapshot();
    initializeWebSocket();
    setInterval(logOrderBook, 500); // Log order book snapshot every 1000ms
}

start();
