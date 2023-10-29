from order_parser import Order_Parser

get_order = Order_Parser()

# order = get_order.order_parser('Can I get a latte  please')
order = get_order.order_parser('Can I get a latte, an americano and a cappucino, please')
print(order)