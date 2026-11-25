from code_navigator.ingestion import code_analysis as ca
from code_navigator.ingestion.code_parser import _regex_defs, parse_code_file
from code_navigator.models import SourceFile


def _sf(path, language, text, kind="code"):
    return SourceFile(path=path, abs_path=path, kind=kind, language=language, size=len(text), sha1="x",
                      text=text, service="svc")


def test_python_units_include_methods_decorators_and_docstrings():
    code = '''import os
from fastapi import FastAPI
app = FastAPI()

@app.post("/orders")
def create_order(req):
    """Create an order."""
    return service.place_order(req)

class OrderService:
    def place_order(self, req):
        self.payments.charge(req)
'''
    units = parse_code_file(_sf("app/main.py", "python", code))
    by_name = {u.name: u for u in units}
    assert "create_order" in by_name and "OrderService.place_order" in by_name
    assert by_name["create_order"].start_line == 5  # decorator included
    assert by_name["create_order"].docstring == "Create an order."
    assert "charge" in by_name["OrderService.place_order"].calls
    assert units[0].kind == "module"


def test_regex_go_methods_get_receiver_type():
    code = '''package payment

func (g *ProviderGateway) ChargeWithRetry(ctx context.Context) error {
\tif x {
\t\treturn nil
\t}
\treturn nil
}

type Handler struct {
\tx int
}
'''
    defs = _regex_defs(code, "go")
    names = {d.name: d for d in defs}
    assert "ProviderGateway.ChargeWithRetry" in names
    assert names["ProviderGateway.ChargeWithRetry"].end == 7
    assert "Handler" in names and names["Handler"].kind == "class"


def test_regex_java_class_and_methods():
    code = '''@RestController
public class ProductController {
    @GetMapping("/{id}")
    public Product get(@PathVariable String id) {
        return service.find(id);
    }
}
'''
    defs = {d.name: d for d in _regex_defs(code, "java")}
    assert "ProductController" in defs
    assert "ProductController.get" in defs
    assert defs["ProductController.get"].start == 2  # annotation line included


def test_markdown_sections():
    text = "# Title\n\nIntro text that is long enough to keep.\n\n## Flow\n\nStep one calls the payment service and more."
    units = parse_code_file(_sf("README.md", "markdown", text, kind="doc"))
    assert [u.name for u in units] == ["Title", "Flow"]


def test_spring_endpoints_use_class_prefix():
    code = '''@RestController
@RequestMapping("/products")
public class ProductController {
    @GetMapping
    public List<Product> list() { return null; }
    @PostMapping("/{id}/reserve")
    public void reserve() {}
}'''
    eps = {(e["method"], e["path"]) for e in ca.extract_endpoints(code, "java")}
    assert ("GET", "/products") in eps
    assert ("POST", "/products/{id}/reserve") in eps
    assert not any(m == "ANY" for m, _ in eps)


def test_go_122_method_patterns_and_flask():
    eps = ca.extract_endpoints('mux.HandleFunc("POST /charge", h.Charge)', "go")
    assert eps[0]["method"] == "POST" and eps[0]["path"] == "/charge"
    eps = ca.extract_endpoints('@app.route("/x", methods=["GET", "POST"])\ndef x(): pass', "python")
    assert eps[0]["method"] == "GET,POST"


def test_topics_and_env_refs():
    py = 'consumer = KafkaConsumer("order-events", "payment-events", bootstrap_servers=os.getenv("KAFKA"))'
    topics = {(t["topic"], t["direction"]) for t in ca.extract_topics(py)}
    assert topics == {("order-events", "sub"), ("payment-events", "sub")}
    go = 'w := &kafka.Writer{Addr: kafka.TCP(b), Topic: "payment-events"}\nw.WriteMessages(ctx, msg)'
    assert ca.extract_topics(go)[0] == {"topic": "payment-events", "direction": "pub", "line": 1}
    java = '@KafkaListener(topics = "order-events", groupId = "x")'
    assert ca.extract_topics(java)[0]["direction"] == "sub"
    refs = [v for v, _ in ca.extract_env_refs('url = os.getenv("PAYMENT_SERVICE_URL")\nprocess.env.ORDER_URL')]
    assert refs == ["PAYMENT_SERVICE_URL", "ORDER_URL"]


def test_express_mounts():
    text = "const ordersRouter = require('./routes/orders');\napp.use('/api/orders', authenticate, ordersRouter);"
    assert ca.express_mounts(text) == [("/api/orders", "./routes/orders")]
