#import os
#import sys
#BACKTESTER_DIR = os.path.realpath(os.path.join(os.getcwd(), '..', '..'))
#sys.path.append(BACKTESTER_DIR)

#from backtester.strategy import Strategy
import enums
from schema import Filter
import pandas as pd
import numpy as np

class backtest:
    def __init__(self, initial_capital=1_000_000, shares_per_contract=100, transaction_fee = 0.5):
        self.initial_capital = initial_capital
        self.shares_per_contract = shares_per_contract
        self.transaction_fee = transaction_fee

        self._options_strategy = None
        self._options_data = None
        
        self.date = None
        self._options_qty = None
        
    @property
    def options_data(self):
        return self._options_data

    @options_data.setter
    def options_data(self, data):
        self._options_data = data
        self._options_schema = data.schema
    
    @property
    def options_strategy(self):
        return self._options_strategy

    @options_strategy.setter
    def options_strategy(self, strat):
        self._options_strategy = strat
        self._exit_thresholds = (None,)*len(strat.legs)
        self.table_columns()

    @property
    def options_qty(self):
        return self._options_qty

    @options_qty.setter
    def options_qty(self, qty):
        self._options_qty = qty

    @property
    def exit_thresholds(self):
        return self._exit_thresholds

    @exit_thresholds.setter
    def exit_thresholds(self, thresh):
        self._exit_thresholds = thresh

    @property
    def open_close_type(self):
        return self._open_close_type

    @open_close_type.setter
    def open_close_type(self, _type):
        '''_type: ('bid','ask')'''
        self._open_close_type = _type

    def run(self, monthly=False):
        self._initialize_inventories()
        self._initialize_trade_log()
        self._initialize_pnl()
        self.current_cash = self.initial_capital
        
        data_iterator = self._data_iterator(monthly)

        t = 0
        for date, options in data_iterator:
            entries = self._execute_entries(date, options)
            exits = self._execute_exits(date, options)
            # print(entries)
            if self._options_inventory.empty:
                pass
            else:
                pnl = self._update_pnl(entries, exits, options, date)

            # print(date,'\n')
            # print(self._portfolio_pnl)
            # print('\n===============================================\n')
            # t += 1
            # if t == 52:
            #     # print(entries)
            #     break 
        self._options_inventory.reset_index(inplace = True, drop = True)
        self._trade_log.reset_index(inplace = True, drop = True)
        return self._trade_log, self._options_inventory, self._portfolio_pnl

    def _signal_fields(self, cost_field):
        fields = {
            self._options_schema['contract']: 'contract',
            self._options_schema['underlying']: 'underlying',
            self._options_schema['expiration']: 'expiration',
            self._options_schema['type']: 'type',
            self._options_schema['strike']: 'strike',
            self._options_schema[cost_field]: 'cost',
            self._options_schema['date']:'date',
            self._options_schema['dte']:'dte',
            self._options_schema['impliedvol']:'impliedvol',
            self._options_schema['delta']:'delta',
            self._options_schema['theta']:'theta',
            self._options_schema['vega']:'vega',
            self._options_schema['gamma']:'gamma',
            'order': 'order',
            'order_type': 'order_type',
            'qty': 'qty'
        }

        return fields



    def _execute_entries(self, date, options):
        entry_signals = []
        open_at = self._open_close_type[0]
        for i, leg in enumerate(self._options_strategy.legs):
            flt = leg.entry_filter
            #cost_field = leg.direction.value # input to specify 

            leg_entries = options[flt(options)]
            if open_at == 'mid':
                leg_entries['mid'] = (leg_entries['bid'] + leg_entries['ask']) / 2
                fields = self._signal_fields('bid')
                fields['mid'] = fields.pop('bid')
            else:
                fields = self._signal_fields(open_at)
                
            leg_entries = leg_entries.reindex(columns=fields.keys())
            leg_entries.rename(columns=fields, inplace=True)

            order = enums.get_order(leg.direction, enums.Signal.ENTRY)
            leg_entries['order'] = order
            leg_entries['order_type'] = 'CR' if (order.value=='STO') else 'DR'

            leg_entries['qty'] = self._options_qty[i]

            if leg.direction == enums.Direction.SELL:
                leg_entries['cost'] = -leg_entries['cost']

            leg_entries['cost'] += self.transaction_fee / self.shares_per_contract
            leg_entries.columns = pd.MultiIndex.from_product([[leg.name], leg_entries.columns])
            entry_signals.append(leg_entries.reset_index(drop=True))
        
        entry_signals = pd.concat(entry_signals, axis=1)
        entry_signals = self._fill_na_leg(entry_signals)

        unit_costs = entry_signals.xs('cost', axis=1, level=1).values
        option_qty = entry_signals.xs('qty', axis=1, level=1).values
        total_costs = np.sum(unit_costs*option_qty)
        totals = pd.DataFrame({'cost': total_costs, 'qty': np.sum(option_qty), 'date': date}, index=[0])
        totals.columns = self.totals_col

        entry_signals = pd.concat([entry_signals,totals], axis=1)
        self._update_trade_log(entry_signals)
        self._update_inventory(entry_signals, signal_name='entry')
        return entry_signals

    def _fill_na_leg(self, signal):
        '''
        signal: 2-leg dataframe
        fill cost and qty of na leg with 0
        '''
        for leg in self._options_strategy.legs:
            #if all columns are na, fill cost and qty with 0 
            if not signal[leg.name].any().sum():
                signal[leg.name, 'cost'] = 0
                signal[leg.name, 'qty'] = 0
        return signal

    def _execute_exits(self, date, options):  
        strategy = self._options_strategy
        current_options_quotes = self._get_current_option_quotes(options)

        filter_masks = []
        close_at=self._open_close_type[1]
        # the loop is to get filter_masks
        for i, leg in enumerate(strategy.legs):
            if self._exit_thresholds[i] is None or self._portfolio_pnl.empty:
                flt = leg.exit_filter
            else:
                profit_thresh, loss_thresh = self._exit_thresholds[i]
                cur_pnl = self._portfolio_pnl[leg.name].iloc[-1]['cum_pnl'] + (abs(current_options_quotes[i].cost[0]) - self._portfolio_pnl[leg.name].iloc[-1]['mark']) \
                            * (-1 if leg.direction == enums.Direction.SELL else 1) - self.transaction_fee / self.shares_per_contract
                cost = abs(self._options_inventory[leg.name].cost[0])
                pct = cur_pnl / cost
                #print(i, cur_pnl, cost)
                if (pct > profit_thresh or pct < loss_thresh) and self._options_inventory[leg.name].notna().values.all():
                    flt = Filter('dte != -1')
                else:
                    flt = leg.exit_filter

            filter_masks.append(flt(current_options_quotes[i]))

            fields = self._signal_fields('bid')
            fields[close_at] = fields.pop('bid')
            
            current_options_quotes[i]=current_options_quotes[i].reindex(columns=fields.keys())
            current_options_quotes[i].rename(columns=fields, inplace=True)
            current_options_quotes[i].columns = pd.MultiIndex.from_product([[leg.name], current_options_quotes[i].columns])
            if not current_options_quotes[i].empty:
                current_options_quotes[i].loc[:, (slice(None), 'order_type')] = 'CR' if (current_options_quotes[i].loc[:, (slice(None), 'order')].values[0][0].value in ['STO', 'STC'])  else 'DR'
            


        exits = []
        for i, leg in enumerate(strategy.legs):
            exit_leg = current_options_quotes[i][leg.name][filter_masks[i]].reset_index(drop=True)
            exit_leg['qty'] = self._options_qty[i]
            if ~leg.direction == enums.Direction.SELL:
                exit_leg['cost'] = -exit_leg['cost']
            exit_leg['cost'] += self.transaction_fee / self.shares_per_contract
            exit_leg.columns = pd.MultiIndex.from_product([[leg.name], exit_leg.columns])
            exits.append(exit_leg)

        exit_signals = pd.concat(exits, axis=1)
        exit_signals = self._fill_na_leg(exit_signals)

        unit_costs = exit_signals.xs('cost', axis=1, level=1).values
        option_qty = exit_signals.xs('qty', axis=1, level=1).values
        total_costs = np.sum(unit_costs*option_qty)
        totals = pd.DataFrame({'cost': total_costs, 'qty': np.sum(option_qty), 'date': date}, index=[0])
        totals.columns = self.totals_col

        exit_signals = pd.concat([exit_signals, totals], axis=1)
        self._update_trade_log(exit_signals)
        self._update_inventory(exit_signals, signal_name='exit')
        
        return exit_signals

    def _get_current_option_quotes(self, options, cost='mid'):
        current_options_quotes = []
        for leg in self._options_strategy.legs:
            #if inventory is empty, set inventory leg to nan
            if self._options_inventory[leg.name]['contract'].empty:
                inventory_leg = pd.DataFrame(np.nan,columns=['contract'],index=[0], dtype='object')
            else:
                inventory_leg = self._options_inventory[leg.name].astype('object')

            leg_options = inventory_leg[['contract']].merge(options,
                                                            how='left',
                                                            left_on='contract',
                                                            right_on=leg.schema['contract'])
            leg_options['order'] = enums.get_order(leg.direction, enums.Signal.EXIT)
            leg_options['mid'] = (leg_options['bid']+leg_options['ask']) / 2
            leg_options['cost'] = leg_options[cost]
            
            if ~leg.direction == enums.Direction.SELL:
                leg_options['cost'] = -leg_options['cost']
            
            current_options_quotes.append(leg_options)
        return current_options_quotes


    def _initialize_inventories(self):
        """Initialize empty options inventories."""
        self._options_inventory = pd.DataFrame(columns=self.inventory_col.append(self.totals_col))

    def _initialize_trade_log(self):
        """Initialize empty trade log."""
        self._trade_log = pd.DataFrame(columns=self.trade_log_col.append(self.totals_col))

    def _initialize_pnl(self):
        """Initialize empty pnl."""
        self._portfolio_pnl = pd.DataFrame(columns=self.portfolio_col.append(self.pnl_totals_col))

    def _update_trade_log(self, signals):
        '''signals: entry/exit signals'''
        if signals.drop('totals',axis=1).drop(['qty','cost'],level=1, axis=1).notna().any().any():
            self._trade_log = self._trade_log.append(signals).reset_index(drop=True)
            self._trade_log = self._trade_log.reindex(columns=self.trade_log_col.append(self.totals_col))


    def _update_inventory(self, signals, signal_name='entry'):
        #if either leg is not NA, update inventory
        if signals.drop('totals',axis=1).drop(['qty','cost'],level=1, axis=1).notna().any().any() and signal_name=='entry':
            if self._options_inventory.empty:
                self._options_inventory = signals
            else:
                for leg in self._options_strategy.legs:
                    #if leg is not NA, update inventory
                    if signals[leg.name].notna().sum().sum() > 2:
                        self._options_inventory[leg.name] = signals[leg.name]

            #compute total costs
            unit_costs = self._options_inventory.drop('totals',axis=1).xs('cost', axis=1, level=1).values
            option_qty = self._options_inventory.drop('totals',axis=1).xs('qty', axis=1, level=1).values
            total_costs = np.sum(unit_costs*option_qty)
            self._options_inventory.loc[0, ('totals','cost')] = total_costs
            self._options_inventory.loc[0,('totals','qty')] = np.sum(option_qty)
            self._options_inventory.loc[0,('totals','date')] = signals['totals','date'].values
            #only keep desired columns
            self._options_inventory = self._options_inventory.loc[:,self.inventory_col.append(self.totals_col)]

        if signals.drop('totals',axis=1).drop(['qty','cost'],level=1, axis=1).notna().any().any() and signal_name=='exit':
            for leg in self._options_strategy.legs:
                if signals[leg.name].notna().sum().sum() > 2:
                    self._options_inventory[leg.name] = np.nan
                    self._options_inventory[leg.name] = self._options_inventory[leg.name].astype('object')
            self._options_inventory = self._fill_na_leg(self._options_inventory)

            unit_costs = self._options_inventory.drop('totals',axis=1).xs('cost', axis=1, level=1).values
            option_qty = self._options_inventory.drop('totals',axis=1).xs('qty', axis=1, level=1).values
            total_costs = np.sum(unit_costs*option_qty)
            self._options_inventory.loc[0, ('totals','cost')] = total_costs
            self._options_inventory.loc[0,('totals','qty')] = np.sum(option_qty)
            self._options_inventory.loc[0,('totals','date')] = signals['totals','date'].values

            self._options_inventory = self._options_inventory.loc[:,self.inventory_col.append(self.totals_col)]

    def _update_pnl(self, entries, exits, options, date):
        """
        Assuming entries price = current options quotes (ie. ignore bid/ask spread on entry day)
        """
        pnl_list = []
        current_options_quotes = self._get_current_option_quotes(options)
        for i, leg in enumerate(self._options_strategy.legs):
            pnl = pd.DataFrame(columns = self.pnl_col)
            current_quotes = current_options_quotes[i]
            if (self._options_inventory[leg.name].isnull().values.any() | self._options_inventory[leg.name].empty):
                # if the leg has not been executed
                if exits[leg.name].isnull().values.any():
                    pnl.columns = pd.MultiIndex.from_product([[leg.name], pnl.columns])
                    pnl.loc[0, :] = 0
                else:
                    leg_pnl = (abs(exits[leg.name].cost) * (-1 if (exits[leg.name].order == enums.Order.BTC).all() else 1) 
                    - self._portfolio_pnl[leg.name]['cost'].values[-1])
                    pnl = current_quotes[self.pnl_col_temp].copy() 
                    pnl.loc[0, :] = 0 
                    pnl['contract'] = self._portfolio_pnl[leg.name]['contract'].values[-1]
                    pnl['expiration'] = self._portfolio_pnl[leg.name]['expiration'].values[-1]
                    pnl['type'] = self._portfolio_pnl[leg.name]['type'].values[-1]
                    pnl['cum_pnl'] = leg_pnl
                    pnl['pnl'] = leg_pnl - self._portfolio_pnl[leg.name]['cum_pnl'].values[-1]
                    pnl['cost'] = self._portfolio_pnl[leg.name]['cost'].values[-1]
                    pnl['qty'] = self._portfolio_pnl[leg.name]['qty'].values[-1]
                    pnl['mark'] = abs(exits[leg.name].cost) - self.transaction_fee / self.shares_per_contract
                    pnl['pct'] = pnl['cum_pnl'] / abs(pnl['cost'])
                    pnl.columns = pd.MultiIndex.from_product([[leg.name], pnl.columns])                
                pnl_list.append(pnl)
            else:
                if ~entries[leg.name].isnull().values.any():
                    # for entries
                    leg_pnl = (abs(current_quotes.cost) * (-1 if (entries[leg.name].order == enums.Order.STO).all() else 1) 
                    - self._options_inventory[leg.name]['cost'])
                else:
                    # for holding
                    leg_pnl = (abs(current_quotes.cost) * (-1 if (self._options_inventory[leg.name].order == enums.Order.STO).all() else 1) 
                    - self._options_inventory[leg.name]['cost'])
                
                pnl = current_quotes[self.pnl_col_temp].copy()
                pnl['cum_pnl'] = leg_pnl
                pnl['pnl'] = leg_pnl if self._portfolio_pnl.empty else leg_pnl - self._portfolio_pnl[leg.name]['cum_pnl'].values[-1]
                pnl['cost'] = self._options_inventory[leg.name]['cost']
                pnl['qty'] = self._options_inventory[leg.name]['qty']
                pnl['mark'] = abs(current_quotes.cost)
                pnl['pct'] = pnl['cum_pnl'] / abs(pnl['cost'])
                pnl.columns = pd.MultiIndex.from_product([[leg.name], pnl.columns])
                pnl_list.append(pnl)
        
        pnl_df = pd.concat(pnl_list, axis = 1)
        pnl_df['totals', 'qty'] = pnl_df.xs('qty', axis=1, level=1, drop_level=False).to_numpy().sum()
        pnl_df['totals', 'pnl'] = pnl_df.xs('pnl', axis=1, level=1, drop_level=False).to_numpy().sum()
        pnl_df['totals', 'cum_pnl'] = pnl_df['totals', 'pnl'] if self._portfolio_pnl.empty else pnl_df['totals', 'pnl'] + self._portfolio_pnl['totals', 'cum_pnl'].values[-1]
        pnl_df['totals', 'date'] = date
        self._portfolio_pnl = self._portfolio_pnl.append(pnl_df)
        self._portfolio_pnl.reset_index(inplace = True, drop = True)


        
    
    def _data_iterator(self, monthly):
        """Returns combined iterator for stock and options data.
        Each step, it produces a tuple like the following:
            (date, stocks, options)

        Returns:
            generator: Daily/monthly iterator over `self._stocks_data` and `self.options_data`.
        """
        if monthly:
            it = self._options_data.iter_months()
        else:
            it = self._options_data.iter_dates()

        return it

    def table_columns(self):
        inventory_col = ['contract','qty','cost', 'date','expiration', 'order']
        trade_log_col = ['contract','qty','cost', 'date','expiration', 
        'dte','order','order_type','impliedvol','delta','theta','vega','gamma']
        self.pnl_col = ['contract', 'type', 'expiration', 'dte', 'impliedvol', 'delta', 'theta', 
        'vega', 'gamma', 'mark', 'cost', 'pct', 'qty', 'pnl', 'cum_pnl']
        self.pnl_col_temp = ['contract', 'type', 'expiration', 'dte', 'impliedvol', 'delta', 
        'theta', 'vega', 'gamma']

        self.inventory_col = pd.MultiIndex.from_product([[l.name for l in self._options_strategy.legs], inventory_col])
        self.totals_col = pd.MultiIndex.from_product([['totals'], ['cost', 'qty', 'date']])
        self.trade_log_col = pd.MultiIndex.from_product([[l.name for l in self._options_strategy.legs], trade_log_col])
        self.portfolio_col = pd.MultiIndex.from_product([[l.name for l in self._options_strategy.legs], self.pnl_col])
        self.pnl_totals_col = pd.MultiIndex.from_product([['totals'], ['qty', 'pnl', 'cum_pnl', 'date']])

    def null_leg(self, leg_name):
        null_leg_col = ['contract','qty','cost', 'date','expiration', 
        'dte','order','order_type','impliedvol','delta','theta','vega','gamma']
        null_leg_col = pd.MultiIndex.from_product([[leg_name], null_leg_col])
        null_leg = pd.DataFrame(np.nan, columns=null_leg_col, index=[0], dtype='object')
        return null_leg

    def trade_log(self, updated_inventory):
        leg_log = updated_inventory.drop(columns='totals', level=0)
        leg_columns = pd.MultiIndex.from_product([leg_log.columns.get_level_values(0).unique(),['contract','underlying','expiration','date','type','strike','cost','order']])
        leg_log = leg_log.reindex(leg_columns, axis=1)

        total_log = updated_inventory[['totals']]
        trade_log = pd.concat([leg_log, total_log], axis=1)
        return trade_log


    def portfolio_value(self):
        entry_cost = []
        for leg in self._options_strategy.legs:
            cost = np.abs(self._trade_log[leg.name].sort_values('date').iloc[0]['cost'])
            entry_cost.append(cost)
        cost_per_order = np.dot(entry_cost, self._options_qty)
        num_of_order = self.initial_capital // (cost_per_order * self.shares_per_contract)

        portfolio_pnl = self._portfolio_pnl[self._portfolio_pnl['totals','qty'] > 0].set_index(('totals','date'))
        portfolio_pnl = portfolio_pnl.drop('totals',axis=1,level=0)

        pnl_per_order = (portfolio_pnl.xs('pnl',level=1, axis=1) * portfolio_pnl.xs('qty',level=1, axis=1)).sum(axis=1)
        pnl_per_order.index.name = 'date'
        pnl_per_day = pnl_per_order * num_of_order * self.shares_per_contract 

        portfolio_value = pnl_per_day.cumsum() + self.initial_capital
        portfolio_value.iloc[0] = self.initial_capital

        return portfolio_value

    def plot_pnl(self):
        pass