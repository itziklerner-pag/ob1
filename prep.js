const params = {
    min_interval: 1000,
    max_interval: 300000,
    fee_maker: 0.02 / 100,
    fee_taker: 0.05 /100,
    tp_margin: 0.05 / 100,
    sl_margin: 0.1 / 100,
    min_margin:  0.1 / 100,
    epsilon: 0.5,
    bid_size: 10000
}
// example of the data :
/*
{"stream":"btcusdt@depth20@100ms","data":{"e":"depthUpdate","E":1713448582480,"T":1713448582479,"s":"BTCUSDT","U":4445146128295,"u":4445146133983,"pu":4445146127899,"b":[["62462.50","1.966"],["62462.20","0.064"],["62462.00","0.002"],["62461.20","0.064"],["62460.60","0.064"],["62460.00","0.002"],["62459.70","0.109"],["62459.60","0.256"],["62459.10","0.002"],["62459.00","0.066"],["62458.60","0.420"],["62458.50","0.779"],["62458.40","0.421"],["62458.10","0.438"],["62458.00","0.066"],["62457.70","0.418"],["62457.60","1.805"],["62457.50","1.473"],["62457.40","1.056"],["62457.00","0.040"]],"a":[["62462.60","2.492"],["62462.70","7.253"],["62462.80","0.067"],["62463.30","0.002"],["62463.50","0.003"],["62463.60","0.002"],["62463.70","0.082"],["62463.80","0.064"],["62464.00","0.002"],["62464.40","0.064"],["62464.50","0.002"],["62465.00","0.015"],["62465.10","0.087"],["62465.40","0.066"],["62466.00","0.081"],["62466.70","0.039"],["62466.90","0.086"],["62467.00","0.064"],["62467.20","0.004"],["62467.50","0.008"]]}},
{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1713448582519,"a":2141769672,"s":"BTCUSDT","p":"62462.50","q":"0.143","f":4905872737,"l":4905872737,"T":1713448582366,"m":true}},
*/
const data = require('./drafts/2024-04-18-06-56-00-depth-1')
// pre process the training data , add a classifier:
// 0 -> no opportunity
// 1 -> buy long opportunity
// 2 -> sell short opportunity
let data_length = data.length, PnL=0

//for (let i=0; i<=data_length; i++) {
data.forEach((d,base_line) => {
    //d=data[i]; base_line = i
    //if(d.stream !== 'btcusdt@depth20@100ms') continue
    let line=0, base_time=d.data.T, 
        base_bid_price=(1-params.fee_taker)* d.data.b[0][0],
        base_ask_price=(1+params.fee_taker)* d.data.a[0][0],
        min_margin=params.min_margin * base_bid_price + params.epsilon,
        stop_loss = base_ask_price * (1-params.tp_margin) ,
        take_profit_long = base_ask_price * (1+params.tp_margin) ,
        take_profit_short = base_bid_price * (1-params.tp_margin) 
    d.Y={}
    while(1){
        line++
        if(line>=data_length) break
        // if(data[line].stream !== 'btcusdt@depth20@100ms') continue
        // if the time is under the minimum interval, even if the price is good , we cant practically to the trade
        if(data[line].data.T - base_time < params.min_interval) continue
        const current_effective_bid = (1-params.fee_maker)*data[line].data.b[0][0]
        const current_effective_ask = (1+params.fee_maker)*data[line].data.a[0][0]
        if(current_effective_bid >= take_profit_long) {
            //PnL+=current_effective_bid - base_ask_price
            d.Y.long=[
                 current_effective_bid - base_ask_price,
                 //PnL,
                 line,
                 data[line].data.T - base_time,
                ]
            //i+=line
            if (d.Y.short) break
        }
        if(current_effective_ask <= take_profit_short) {
            d.Y.short=[
                 base_bid_price - current_effective_ask,
                 //PnL,
                 line,
                 data[line].data.T - base_time
                ]
            //i+=line
            if (d.Y.long) break
        }
        // time based stop loss
        if(data[line].data.T - base_time > params.max_interval) {
            if (d.Y.short || d.Y.long) break
            //PnL+=current_effective_bid - base_ask_price
            d.Y.hold = [
                 current_effective_bid - base_ask_price,
                 PnL,
                 line,
                 data[line].data.T - base_time
                ]
            //i+=line
            break
        }
        // price based stop loss
            
        if(0 && current_effective_bid <= stop_loss) {
            PnL+=current_effective_bid - base_ask_price
            d.Y=[3,
                    current_effective_bid - base_ask_price,
                    PnL,
                    line,
                    data[line].data.T - base_time
                ]
            i+=line
            break
        }
        
    }

    console.log(base_line,JSON.stringify(d))
})

