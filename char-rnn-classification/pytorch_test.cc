#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  at::Tensor inputs = torch::rand({1,58});

  assert(module != nullptr);
  std::cout << "ok\n";

  at::Tensor hidden = torch::zeros({1,128});
  torch::jit::IValue output = module->forward({inputs, hidden});
  std::cout << typeid(output).name() << std::endl;

  auto elems =  output.toTuple().get()->elements(); // toTensor()
  std::cout << elems[0].toTensor() << std::endl;
  //std::cout << output.slice(1, 0, 5) << std::endl;

  return 0;
}