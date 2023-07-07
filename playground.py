import re
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from pyee import EventEmitter
from streamlit.type_util import is_iterable

ACCOUNTS = ["alice", "bob", "charlie", "david", "eve", "frank", "grace", "heidi", "ivan", "judy", "kris", "lucy", "mike", "niaj", "oscar", "peggy", "rupert", "sybil", "trent", "victor", "wendy"]


@dataclass
class CoinInfo:
    id: str
    symbol: str
    name: str
    categories: List[str]
    description: Dict[str, str]
    image: Dict[str, str]
    last_updated: str


class TokenFeature(Enum):
    ERC20 = 1
    LIQUIDITY_TOKEN = 2


class WildcardEventEmitter(EventEmitter):
    def emit(self, event, *args, **kwargs):
        # Call handlers for the specific event
        # super().emit(event, *args, **kwargs)
        # Call handlers for all events
        super().emit("*", event, *args, **kwargs)


class Token(WildcardEventEmitter):
    def __init__(self, symbol: str, name: str, feature: TokenFeature = TokenFeature.ERC20, pool: Optional["Pool"] = None, coin_info: Optional[CoinInfo] = None, market_price: float = 0):
        super().__init__()
        self.balances: Dict[str, float] = {}
        self.total_supply = 0
        self.feature = feature
        self.symbol = symbol
        self.name = name
        self.pool = pool
        self.coin_info = coin_info
        self.market_price = market_price

    @staticmethod
    def from_coin_info(coin_info: CoinInfo):
        return Token(symbol=coin_info.symbol.upper(), name=coin_info.name, feature=TokenFeature.ERC20, coin_info=coin_info)

    def set_market_price(self, price: float):
        self.market_price = price
        self.emit("New Price", {"price": price})

    def mint(self, amount: float, to: str):
        self.total_supply += amount
        self.balances[to] = self.balances.get(to, 0) + amount
        self.emit("Minted", {"user": to, self.symbol: amount})

    def transfer(self, from_: str, to: str, amount: float):
        if self.balance_of(from_) < amount:
            raise ValueError(f"No sufficient funds ({amount} > {self.balance_of(from_)})")
        self.balances[from_] -= amount
        self.balances[to] = self.balances.get(to, 0) + amount
        # self.emit("Transferred", {"from": from_, "to": to, "amount": amount})

    def burn(self, from_: str, amount: float):
        self.balances[from_] -= amount
        self.total_supply -= amount
        self.emit("Burnt", {"user": from_, self.symbol: amount})

    def balance_of(self, from_: str) -> float:
        return self.balances.get(from_, 0)

    def share_of(self, from_: str) -> float:
        return self.balance_of(from_) / self.total_supply


class Pool(WildcardEventEmitter):
    def __init__(self, address: str, token1: Token, token2: Token, fee_percent: float = 0.0):
        super().__init__()
        self.token1 = token1
        self.token2 = token2
        self.account = address
        self.fee_rate = fee_percent / 100
        self.collected_fees = {token1.symbol: 0, token2.symbol: 0}
        self.pool_token = Token(symbol=f"{token1.symbol}|{token2.symbol}", name=f"{token1.symbol} {token2.symbol} Pool Shares", feature=TokenFeature.LIQUIDITY_TOKEN, pool=self)
        self.pool_token.pool = self

    def add_liquidity(self, sender: str, amount1: float, amount2: float):
        self.token1.transfer(sender, self.account, amount1)
        self.token2.transfer(sender, self.account, amount2)
        self.pool_token.mint(sqrt(amount1 * amount2), sender)
        self.emit("LP Add", {"user": sender, self.token1.symbol: amount1, self.token2.symbol: amount2})

    def remove_liquidity(self, sender: str, amount: float):
        share = amount / self.pool_token.total_supply

        if self.pool_token.balance_of(sender) < amount:
            raise ValueError("Not enough liquidity tokens on your account")

        share = min(share, 1)
        withdraw1 = share * self.token1.balance_of(self.account)
        withdraw2 = share * self.token2.balance_of(self.account)

        self.token1.transfer(self.account, sender, withdraw1)
        self.token2.transfer(self.account, sender, withdraw2)
        self.pool_token.burn(sender, amount)
        self.emit("LP Remove", {"user": sender, self.token1.symbol: -withdraw1, self.token2.symbol: -withdraw2})

    def k(self):
        return self.reserves()[0] * self.reserves()[1]

    def quote(self, from_token: Token, to: Token, from_amount: float) -> float:
        from_reserve = from_token.balance_of(self.account)
        to_reserve = to.balance_of(self.account)
        quoted_amount = to_reserve - self.k() / (from_amount - self.fee_rate * from_amount + from_reserve)
        if from_amount != 0:
            price = max(quoted_amount / from_amount, from_amount / quoted_amount)
            self.emit("quote", {"price": price, from_token.symbol: from_amount, to.symbol: quoted_amount})
        return quoted_amount

    def swap(self, sender: str, from_token: Token, to_token: Token, amount: float):
        quoted_amount = self.quote(from_token, to_token, amount)
        self.collected_fees[from_token.symbol] += int(self.fee_rate * amount)

        from_token.transfer(sender, self.account, amount)
        to_token.transfer(from_=self.account, to=sender, amount=quoted_amount)
        self.emit("Swap", {"user": sender, from_token.symbol: -amount, to_token.symbol: quoted_amount})
        self.emit("NewReserves", {self.token1.symbol: self.token1.balance_of(self.account), self.token2.symbol: self.token2.balance_of(self.account)})

        # adjust price of more expensive token
        if quoted_amount > amount:
            from_token.set_market_price(self.price(from_token))
        else:
            to_token.set_market_price(self.price(to_token))

        return quoted_amount

    def tokens(self) -> List[str]:
        return [self.token1.symbol, self.token2.symbol]

    def reserves(self) -> List[float]:
        return [self.token1.balance_of(self.account), self.token2.balance_of(self.account)]

    def pool_info(self) -> Dict:
        return {"name": self.account, "K": self.k(), "LP Tokens": self.pool_token.total_supply, "fee": self.fee_rate}

    def price(self, token: Token) -> float:
        other = self.token1 if token == self.token2 else self.token2
        return other.balance_of(self.account) / token.balance_of(self.account)

    def prices(self) -> List[float]:
        bal1 = self.token1.balance_of(self.account)
        bal2 = self.token2.balance_of(self.account)
        return [bal2 / bal1, bal1 / bal2]

    def fees(self) -> List[float]:
        fee1 = self.collected_fees[self.token1.symbol]
        fee2 = self.collected_fees[self.token2.symbol]
        return [fee1, fee2]


class EventLogger:
    def __init__(self):
        self.logs = []

    def attach(self, emitters):
        if not is_iterable(emitters):
            emitters = [emitters]
        for emitter in emitters:
            # Attach a listener to the 'new_listener' event
            emitter.on("*", self.log_event)

    def log_event(self, *args, **kwargs):
        if args is None:
            return
        name = args[0]
        event = args[1]
        self.logs.append(event | {"event": name})

    def write_log(self, event_filter):
        if len(self.logs) == 0:
            return
        log_df = pd.DataFrame(self.logs)
        # Perform wildcard matching
        filters = [filter_.strip() for filter_ in event_filter.split(",")]
        any_exclude = any(filter_.startswith("-") for filter_ in filters)
        matched_rows = pd.Series([any_exclude is True] * len(log_df))
        for filter_ in filters:
            exclude_mode = False
            if filter_.startswith("-"):
                exclude_mode = True
                filter_ = filter_[1:]
            # Create the regular expression pattern with the re.IGNORECASE flag
            pattern = f"(?i).*{re.escape(filter_)}.*"
            # Filter the DataFrame based on the search expression and exclude mode
            new_matched_rows = log_df["event"].str.contains(pattern, regex=True)
            if exclude_mode:
                new_matched_rows = ~new_matched_rows
            if any_exclude:
                matched_rows = matched_rows & new_matched_rows
            else:
                matched_rows = matched_rows | new_matched_rows
        log_df = log_df.loc[matched_rows, :]
        log_df = log_df.iloc[::-1]
        st.dataframe(log_df, height=800, use_container_width=True)


def pool_for_token(pools, token):
    return next((p for p in pools if p.pool_token == token), None)


def pool_shares(pool, address):
    share = pool.pool_token.share_of(address)
    return [
        pool.token1.balance_of(pool.account) * share,
        pool.token2.balance_of(pool.account) * share,
    ]


def compute_pool_share_usd_value(pool, address):
    shares = pool_shares(pool, address)
    return shares[0] * pool.token1.market_price + shares[1] * pool.token2.market_price


def compute_usd_value(address, tokens, pools):
    all_vals = [compute_pool_share_usd_value(pool_for_token(pools, t), address) if t.feature == TokenFeature.LIQUIDITY_TOKEN else t.balance_of(address) * t.market_price for t in tokens]
    return sum(all_vals)


def set_defaults():
    logger = EventLogger()

    eth = Token("ETH", "Ethereum", TokenFeature.ERC20, market_price=2000)
    dai = Token("DAI", "Dai", TokenFeature.ERC20, market_price=1)

    logger.attach(eth)
    logger.attach(dai)

    tokens = [eth, dai]

    eth_dai_pool = Pool(address="HyperPool", token1=eth, token2=dai)

    eth.mint(100, "alice")
    dai.mint(200_000, "alice")

    eth.mint(10, "bob")
    dai.mint(10_000, "bob")

    dai.mint(20_000, "lucy")
    eth.mint(10, "lucy")
    eth_dai_pool.add_liquidity("lucy", 10, 20_000)

    pools = [eth_dai_pool]
    logger.attach(eth_dai_pool)

    accounts = ["alice", "bob"]

    event_filter = ""

    return {"event_filter": event_filter, "logger": logger, "tokens": tokens, "pools": pools, "accounts": accounts, "eth_amount": 0.0, "dai_amount": 0.0}


def hr(col1, col2):
    with col1, col2:
        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


def reset():
    st.session_state = set_defaults()


def app():
    st.set_page_config(page_icon="💰", layout="wide")
    st.header("AMM App")
    st.button(label="Reset", on_click=reset)

    if "tokens" not in st.session_state:
        reset()

    col1, col2, col3 = st.columns([2.6, 2.4, 4])

    with col1:
        selected_account = st.radio("Select an account", st.session_state["accounts"])
    with col2:
        selected_action = st.radio("Select an action", ["Swap", "Add liquidity", "Remove liquidity"])
    with col3:
        st.session_state["event_filter"] = st.text_input("Event filter (case insensitive, comma separated, ex: `-q` `-q,-m`, `swap`, `swap,LP`)", value="")
        st.session_state["logger"].write_log(st.session_state["event_filter"])
    selected_pool: Pool = st.session_state["pools"][0]  # hardcode to ETH-DAI pool

    with col1:
        if selected_action == "Swap":
            from_token_symbol = st.radio("Sell", [token.symbol for token in st.session_state["tokens"]])
            from_token = next(token for token in st.session_state["tokens"] if token.symbol == from_token_symbol)
            from_token_index = [i for i, t in enumerate(st.session_state["tokens"]) if t.symbol == from_token.symbol][0]
            to_token = st.session_state["tokens"][1 - from_token_index]
            from_amount = st.number_input(f"{from_token_symbol} Amount", min_value=0.0, value=1.00, format="%f")
            market_price = from_token.market_price / to_token.market_price
            from_value = from_amount * market_price
            quote_amount = selected_pool.quote(from_token, to_token, from_amount)
            quoted_price = quote_amount / from_value * market_price
            slippage = (quote_amount - from_value) / (from_value) if from_value != 0 else 0
            st.write(f"For {quote_amount} {to_token.symbol}")
            st.write(f"Price: {quoted_price} {to_token.symbol}")
            st.write(f"\nSlippage of {slippage:.2%}%")
            st.button(label="Swap", on_click=selected_pool.swap, args=(selected_account, from_token, to_token, from_amount))
        else:
            chosen_token_symbol = st.radio("Choose Token", [token.symbol for token in st.session_state["tokens"]])
            chosen_token = next(token for token in st.session_state["tokens"] if token.symbol == chosen_token_symbol)
            chosen_token_index = [i for i, t in enumerate(st.session_state["tokens"]) if t.symbol == chosen_token.symbol][0]
            other_token = st.session_state["tokens"][1 - chosen_token_index]
            chosen_amount = st.number_input(f"{chosen_token_symbol} Amount", min_value=0.0, value=1.00, format="%f")
            market_price = chosen_token.market_price / other_token.market_price
            chosen_value = chosen_amount * market_price
            other_amount = chosen_value / other_token.market_price
            st.write(f"and {other_amount} {other_token.symbol}")

            amounts = []
            for token in st.session_state["tokens"]:
                if token.symbol == chosen_token_symbol:
                    amounts.append(chosen_amount)
                if token.symbol == other_token.symbol:
                    amounts.append(other_amount)
            args = [selected_account] + amounts

            if selected_action == "Add liquidity":
                st.button(label="Add liquidity", on_click=selected_pool.add_liquidity, args=args)
            elif selected_action == "Remove liquidity":
                st.button(label="Remove liquidity", on_click=selected_pool.remove_liquidity, args=args)

    with col2:
        st.markdown("<center>", unsafe_allow_html=True)
        if st.session_state["pools"]:
            st.write(f"Pool{'s' if len(st.session_state['pools']) > 1 else ''}")
            for pool in st.session_state["pools"]:
                markdown_text = ""
                for k, v in pool.pool_info().items():
                    if k in ["reserves", "prices"]:
                        continue
                    if isinstance(v, int):
                        formatted_v = f"{v:,.0f}"
                    else:
                        formatted_v = v
                    # st.write(f"{k}: {formatted_v}")
                    markdown_text += f"{k}: {formatted_v}<br>"
                st.markdown(markdown_text, unsafe_allow_html=True)
                token_df = pd.DataFrame(index=pool.tokens(), data={"reserves": pool.reserves(), "price": pool.prices(), "fees": pool.fees()})
                st.dataframe(token_df, use_container_width=True)

    with col1:
        for account in st.session_state["accounts"]:
            separator = "\n- "
            token_balances = separator.join([f" {token.symbol}: {token.balance_of(account):,.2f}" for token in st.session_state["tokens"]])
            st.markdown(f"{account}{separator}{token_balances}{separator}${compute_usd_value(account, st.session_state['tokens'], st.session_state['pools']):,.0f}")


if __name__ == "__main__":
    app()
