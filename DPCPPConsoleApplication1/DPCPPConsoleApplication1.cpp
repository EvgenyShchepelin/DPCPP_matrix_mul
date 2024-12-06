#include <CL/sycl.hpp>
#include <iostream>
using namespace std;
using namespace sycl;
class A_init;
class B_init;
class C_init;
class CAB_calc;
const int Matrix_size = 4096;

double GetExecutionTime(const event& e){
  double start = e.get_profiling_info<info::event_profiling::command_start>();
  double end = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end - start) * 1e-9;
  return kernel_time;
}

int main(){
  try{
    sycl::queue queue(cpu_selector_v, sycl::property::queue::enable_profiling {});
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    sycl::buffer<float, 2> Matrix_A(sycl::range(Matrix_size, Matrix_size));
    sycl::buffer<float, 2> Matrix_B(sycl::range(Matrix_size, Matrix_size));
    sycl::buffer<float, 2> Matrix_C(sycl::range(Matrix_size, Matrix_size));
        
    auto Event_A = queue.submit([&](sycl::handler& handler){
      auto accessor = Matrix_A.get_access<sycl::access::mode::write>(handler);

      handler.template parallel_for<A_init>(sycl::range(Matrix_size, Matrix_size), [=](sycl::id<2> index){
        accessor[index] = 5.234f;
      });
    });
     
    auto Event_B = queue.submit([&](sycl::handler& handler){
      auto accessor = Matrix_B.get_access<sycl::access::mode::write>(handler);

      handler.template parallel_for<B_init>(sycl::range(Matrix_size, Matrix_size), [=](sycl::id<2> index){
        accessor[index] = 1.234f;
      });
    });

    auto Event_C = queue.submit([&](sycl::handler& handler){
      auto accessor = Matrix_C.get_access<sycl::access::mode::write>(handler);

      handler.template parallel_for<C_init>(sycl::range(Matrix_size, Matrix_size), [=](sycl::id<2> index){
        accessor[index] = 0.0f;
      });
    });
      
    auto Event_CAB = queue.submit([&](sycl::handler& handler){
      handler.depends_on({Event_A, Event_B, Event_C});
      auto matrixA = Matrix_A.get_access<sycl::access::mode::read>(handler);
      auto matrixB = Matrix_B.get_access<sycl::access::mode::read>(handler);
      auto matrixC = Matrix_C.get_access<sycl::access::mode::write>(handler);
      const int tileSize = 64;
      sycl::range group_size{64, 64};

      handler.template parallel_for_work_group<CAB_calc>({64, 64}, group_size, [=](sycl::group<2> group){
        float tileA[64];

        for (int kk = 0; kk < Matrix_size; kk += tileSize){
          group.parallel_for_work_item([&](sycl::h_item<2> item){
            int m = item.get_global_id()[0];
            int i = item.get_local_id()[1];
            tileA[i] = matrixA[m][kk + i];
          });

          group.parallel_for_work_item([&](sycl::h_item<2> item){
            int m = item.get_global_id()[0];
            int n = item.get_global_id()[1];

            for (int k = 0; k < tileSize; k++)
              matrixC[m][n] += tileA[k] * matrixB[kk + k][n];
          });
        }
      });
    });

    std::cout << "Matrix_A initialize speed: " << GetExecutionTime(Event_A) << " second \n";
    std::cout << "Matrix_B initialize speed: " << GetExecutionTime(Event_B) << " second \n";
    std::cout << "Matrix_C initialize speed: " << GetExecutionTime(Event_C) << " second \n";
    std::cout << "Calculating C = A * B speed: " << GetExecutionTime(Event_CAB) << " second \n";
  }
  catch (sycl::exception const& e){
    std::cout << "exception\n";
    terminate();
  }
}