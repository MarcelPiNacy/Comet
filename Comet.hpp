/*
	Copyright 2022 Marcel Pi Nacy
	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at
		http://www.apache.org/licenses/LICENSE-2.0
	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/

#ifndef COMET_INCLUDED
#define COMET_INCLUDED



#if ((defined(__clang__) || defined(__GNUC__)) && !defined(NDEBUG)) || (defined(_MSVC_LANG) && defined(_DEBUG))
	#define COMET_DEBUG
#endif



#ifndef COMET_DEBUG

	#ifdef __has_attribute

		#if __has_attribute(always_inline)
			#define COMET_INLINE __attribute__((always_inline))
		#else
			#define COMET_INLINE
		#endif

		#if __has_attribute(flatten)
			#define COMET_FLATTEN __attribute__((flatten))
		#else
			#define COMET_FLATTEN
		#endif

		#if __has_attribute(noinline)
			#define COMET_NOINLINE __attribute__((noinline))
		#else
			#define COMET_NOINLINE
		#endif

	#elif defined(_MSVC_LANG)
		#define COMET_INLINE __forceinline
		#define COMET_FLATTEN COMET_INLINE
		#define COMET_NOINLINE __declspec(noinline)
	#else
		#define COMET_INLINE
		#define COMET_FLATTEN
		#define COMET_NOINLINE
	#endif

#else

	#define COMET_INLINE inline
	#define COMET_FLATTEN
	#define COMET_NOINLINE

#endif



#ifdef __has_builtin

	#if __has_builtin(__builtin_debugtrap) && !defined(COMET_DEBUG)
		#define COMET_DEBUGTRAP __builtin_debugtrap()
	#else
		#define COMET_DEBUGTRAP
	#endif
	
	#if __has_builtin(__builtin_unreachable)
		#define COMET_UNREACHABLE __builtin_unreachable()
	#else
		#define COMET_UNREACHABLE abort()
	#endif
	
	#if __has_builtin(__builtin_ia32_pause)
		#define COMET_SPIN __builtin_ia32_pause()
	#else
		#define COMET_SPIN 
	#endif
	
	#if __has_builtin(__builtin_expect)
		#define COMET_LIKELY_IF(...) if (__builtin_expect((__VA_ARGS__), 1))
		#define COMET_UNLIKELY_IF(...) if (__builtin_expect((__VA_ARGS__), 0))
	#else
		#define COMET_LIKELY_IF(...) if ((__VA_ARGS__))
		#define COMET_UNLIKELY_IF(...) if ((__VA_ARGS__))
	#endif

#elif defined(_MSVC_LANG)

	#define COMET_DEBUGTRAP __debugbreak()
	#define COMET_UNREACHABLE __assume(0)
	#define COMET_SPIN
	#define COMET_LIKELY_IF(...) if ((__VA_ARGS__))
	#define COMET_UNLIKELY_IF(...) if ((__VA_ARGS__))

#endif

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <string_view>

#undef Yield






#ifndef COMET_SLEEP_ON_IDLE
#define COMET_SLEEP_ON_IDLE 0
#endif

#ifndef COMET_MAX_PRIORITY
#define COMET_MAX_PRIORITY 4
#endif

#ifndef COMET_FIXED_PROCESSOR_COUNT
#define COMET_FIXED_PROCESSOR_COUNT 0
#endif






namespace Comet
{
	enum class SchedulerState : uint8_t
	{
		Uninitialized,
		Initializing,
		Running,
		Pausing,
		Paused,
		Resuming,
		ShuttingDown,
		Finalizing,

		MaxEnum,
	};



	enum class DispatchResult : uint8_t
	{
		Success,
		Sequential,
		Failure,

		MaxEnum,
	};



	enum class DispatchErrorPolicy : uint8_t
	{
		RunSequentially,
		Spin,
		Return,
		Abort,

		MaxEnum,
	};



	enum class Implementation : uint8_t
	{
		Unimplemented,
		WindowsFibers,
		SetContext,

		MaxEnum,
	};



	constexpr Implementation GetImplementation()
	{
		return Implementation::
#ifdef _WIN32
			WindowsFibers;
#else
			Unimplemented;
#endif
	};



	constexpr std::string_view GetImplementationName()
	{
		return
#ifdef _WIN32
			"Windows Fibers";
#else
			"N/A";
#endif
	};



	class Fence
	{
		uint32_t state;

	public:

		constexpr Fence() :
			state(UINT32_MAX)
		{
		}

		~Fence() = default;

		void Signal();

		void Await();

		void Reset();
	};



	class Counter
	{
		uint64_t x, y;
	public:

		constexpr Counter(uint64_t value) :
			x(value), y(UINT64_MAX)
		{
		}

		~Counter() = default;

		bool Decrement();
		bool WakeAll();
		bool Await();
		void Reset(uint64_t new_value);
		uint64_t Value() const;
	};



	class Mutex
	{
		uint64_t state;
	public:

		void Lock();
		bool IsLocked() const;
		bool TryLock();
		void Unlock();
	};



	class MutexMCS
	{
		uintptr_t state;
	public:

		struct Node
		{
			Node* next = nullptr;
			uint32_t index = UINT32_MAX;
		};



		constexpr MutexMCS() :
			state()
		{
		}

		~MutexMCS() = default;

		void Lock(Node& node);
		bool IsLocked() const;
		bool TryLock(Node& node);
		void Unlock(Node& node);
	};



	using DebugMessageFn = void(*)(void* context, const char* message, uint32_t size);
		
	struct DebugOptions
	{
		void* context;
		DebugMessageFn info;
		DebugMessageFn warning;
		DebugMessageFn error;
	};

	struct InitOptions
	{
		static InitOptions Default();

		uint32_t thread_stack_size;
		uint32_t task_stack_size;
#if COMET_FIXED_PROCESSOR_COUNT == 0
		uint32_t max_threads;
#endif
		uint32_t max_tasks;
		uint64_t reseed_threshold_ns;
		const uint32_t* affinity_indices;
		const DebugOptions* debug;
	};



	struct TaskOptions
	{
		const uint32_t* preferred_thread;
		Counter* counter;
		DispatchErrorPolicy error_policy;
		uint8_t priority;

		static constexpr TaskOptions Default() { return {}; }
	};



	bool Init(const InitOptions& options);

	COMET_INLINE bool Init()
	{
		return Init(InitOptions::Default());
	}

	bool Pause();
	bool Resume();
	bool Shutdown();
	bool TryFinalize();
	bool Finalize();
	SchedulerState GetSchedulerState();
	bool IsTask();
	DispatchResult Dispatch(void(*fn)(void* arg), void* arg);
	DispatchResult Dispatch(void(*fn)(void* arg), void* arg, TaskOptions options);
	void Yield();
	void Exit();
	uint64_t ThisTaskID();
	uint32_t WorkerThreadIndex();
	uint32_t MaxTasks();



#if COMET_FIXED_PROCESSOR_COUNT == 0
	uint32_t WorkerThreadCount();
#else
	constexpr uint32_t WorkerThreadCount() { return COMET_FIXED_PROCESSOR_COUNT; }
#endif



	namespace Debug
	{
#ifdef COMET_DEBUG
		void SetYieldTrap(bool k);
#else
		constexpr void SetYieldTrap(bool value) {}
#endif
	}



	namespace RCU
	{
		uint32_t GetSchedulerSnapshotSize();
		void GetSchedulerSnapshot(void* out);
		uint32_t TrySync(void* snapshot, uint32_t prior_result = 0);

		template <typename F>
		COMET_INLINE void CriticalSection(F&& body)
		{
			Debug::SetYieldTrap(true);
			body();
			Debug::SetYieldTrap(false);
		}
	}



	template <typename T, typename F>
	COMET_INLINE void CriticalSection(T& lock, F&& body)
	{
		lock.Lock();
		body();
		lock.Unlock();
	}



	template <typename F>
	COMET_INLINE void CriticalSection(MutexMCS& lock, F&& body)
	{
		MutexMCS::Node node;
		lock.Lock(node);
		body();
		lock.Unlock(node);
	}



	template <typename F>
	COMET_INLINE auto Dispatch(F&& fn, TaskOptions options = {})
	{
		struct Context
		{
			F fn;
			Fence fence;
		};
		
		Context c = { std::forward<F>(fn), Fence() };

		auto code = Dispatch([](void* ptr)
		{
			auto& ctx_ref = *(Context*)ptr;
			F fn = std::move(ctx_ref.fn);
			ctx_ref.fence.Signal();
			fn();
		}, &c, options);

		switch (code)
		{
		case DispatchResult::Success:
			c.fence.Await();
			c.fence.Reset();
			break;
		case DispatchResult::Sequential:
		case DispatchResult::Failure:
			break;
		default:
			COMET_UNREACHABLE;
		}

		return code;
	}



	template <typename I, typename J>
	COMET_INLINE bool ForEach(I begin, J end, void(*body)(I value), TaskOptions options = {})
	{
		using F = decltype(body);
		
		COMET_UNLIKELY_IF(begin == end)
			return true;
		
		assert(end > begin);
		
		struct Context
		{
			F fn; I it; Fence fence;
		};

		Context c = { body, begin, Fence() };

		for (; c.it < end; ++c.it)
		{
			auto code = Dispatch([](void* ptr)
			{
				auto& ctx_ref = *(Context*)ptr;
				auto fn = ctx_ref.fn;
				auto it = ctx_ref.it;
				ctx_ref.fence.Signal();
				fn(it);
			}, &c, options);

			switch (code)
			{
			case DispatchResult::Success:
				c.fence.Await();
				c.fence.Reset();
				break;
			case DispatchResult::Sequential:
				break;
			case DispatchResult::Failure:
				return false;
			default:
				COMET_UNREACHABLE;
			}
		}
		return true;
	}



	namespace Detail
	{
		template <typename T>
		constexpr bool is_lambda = !(
			std::is_trivially_constructible_v<T> &&
			std::is_trivially_copyable_v<T> &&
			std::is_trivially_destructible_v<T> &&
			std::is_class_v<T> &&
			std::is_empty_v<T>);
	}



	template <typename I, typename J, typename F>
	// requires(Detail::is_lambda<F>)
	bool ForEach(I begin, J end, F&& body, TaskOptions options = {})
	{
		COMET_UNLIKELY_IF(begin == end)
			return true;

		assert(end > begin);
		assert(options.counter == nullptr);

		struct Context
		{
			F fn;
			I it;
			Fence fence;
		};
		
		Context c = { std::forward<F>(body), begin, Fence() };
		Counter ctr = end - begin;
		
		options.counter = &ctr;

		for (; c.it < end; ++c.it)
		{
			auto code = Dispatch([](void* ptr)
			{
				auto& ctx = *(Context*)ptr;
				I it = ctx.it;
				ctx.fence.Signal();
				ctx.fn(it);
			}, &c, options);

			switch (code)
			{
			case DispatchResult::Success:
				c.fence.Await();
				c.fence.Reset();
				break;
			case DispatchResult::Sequential:
				break;
			case DispatchResult::Failure:
				return false;
			default:
				COMET_UNREACHABLE;
			}
		}

		ctr.Await();
		return true;
	}
}
#endif



#define COMET_IMPLEMENTATION



#ifdef COMET_IMPLEMENTATION

#ifdef __has_attribute
#if __has_attribute(naked)
#define COMET_NAKED __attribute__((naked))
#else
#define COMET_NAKED
#endif
#elif defined(_MSVC_LANG)
#define COMET_NAKED __declspec(naked)
#endif

#include <cstddef>
#include <utility>
#include <atomic>
#include <bit>

#ifdef COMET_CACHE_LINE_SIZE
#define COMET_SHARED_ATTR alignas(COMET_CACHE_LINE_SIZE)
#else
#include <new>
#define COMET_SHARED_ATTR alignas(std::hardware_destructive_interference_size)
#endif

#include <format> // Some platforms still lack this header, for some reason (FreeBSD 13)

#ifdef __FreeBSD__
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <pthread_np.h>
#define COMET_TASK_CALL
#define COMET_THREAD_CALL
#elif defined(_WIN32)
#include <Windows.h>
#undef Yield
#endif


#ifndef COMET_FORMAT_NAMESPACE
#endif


namespace Comet
{
	using std::atomic;

	template <typename T>
	COMET_INLINE static T& NonAtomicRef(atomic<T>& source)
	{
		static_assert(atomic<T>::is_always_lock_free);
		return *(T*)&source;
	}

	COMET_INLINE static uint64_t WellonsMix64(uint64_t x)
	{
		x ^= x >> 32;
		x *= UINT64_C(0xd6e8feb86659fd93);
		x ^= x >> 32;
		x *= UINT64_C(0xd6e8feb86659fd93);
		x ^= x >> 32;
		return x;
	}

	template <typename T>
	COMET_INLINE static uint64_t ReduceMix(T value)
	{
		auto x = (uint64_t)value;
		x ^= std::rotr(x, x & 63);
		return (uint64_t)x;
	}

	template <typename T, typename... U>
	COMET_INLINE static uint64_t ReduceMix(T value, U... remaining)
	{
		return value ^ ReduceMix(remaining...);
	}

	struct TaskContext;
	struct ThreadContext;

#ifdef __FreeBSD__

#define COMET_TASK_CALL
#define COMET_THREAD_CALL

	using ThreadHandle = pthread_t;
	using ThreadParam = void*;
	using ThreadReturn = void*;
	using ThreadEntryPoint = ThreadReturn(*)(ThreadParam);
	using TaskEntryPoint = void(*)(void*);

#elif defined(_WIN32)

#define COMET_TASK_CALL __stdcall
#define COMET_THREAD_CALL __stdcall

	using ThreadHandle = HANDLE;
	using ThreadParam = PVOID;
	using ThreadReturn = DWORD;
	using ThreadEntryPoint = LPTHREAD_START_ROUTINE;
	using TaskEntryPoint = LPFIBER_START_ROUTINE;

#endif

#ifdef _WIN32
	using Context = HANDLE;
#else
#endif

	struct TaskContext
	{
		Context context;
		void(*fn)(void* arg);
		void* arg;
		Counter* counter;
		uint32_t generation;
		uint32_t next;
		uint8_t priority;
		uint8_t sleeping;
	};

	struct MPSCQueue
	{
		COMET_SHARED_ATTR atomic<uint32_t> head;
		COMET_SHARED_ATTR atomic<uint32_t> size;
		COMET_SHARED_ATTR atomic<uint32_t>* values;
		uint32_t tail;
	};

	struct ThreadContext
	{
		Context main_context;
		uint64_t romu2jr[2];
		uint64_t next_reseed;
		uint64_t local_accumulator;
		TaskContext* this_task;
		uint32_t yield_count;
		uint32_t reseed_counter;
		uint32_t spinlock_next;
		uint8_t quit_flag;
#if COMET_SLEEP_ON_IDLE
		uint32_t last_counter;
#endif
		bool spinlock_flag;
#ifdef COMET_DEBUG
		bool yield_trap;
#endif
#if COMET_SLEEP_ON_IDLE
		COMET_SHARED_ATTR atomic<uint32_t> counter;
#endif
		COMET_SHARED_ATTR MPSCQueue queues[COMET_MAX_PRIORITY];
		ThreadHandle handle;
	};

	struct alignas(8) IndexPair
	{
		uint32_t first, second;
	};

	struct COMET_SHARED_ATTR TaskFreeList
	{
		atomic<IndexPair> head;
	};

	struct CounterState
	{
		std::atomic<uint64_t> counter;
		std::atomic<IndexPair> queue;
	};

	static thread_local ThreadContext* this_thread;

	enum class TaskState : uint8_t
	{
		Uninitialized,
		Running,
		Pausing,
		Paused,
		Resuming,
	};

#if COMET_FIXED_PROCESSOR_COUNT == 0
	static ThreadContext* thread_contexts;
#else
	static ThreadContext thread_contexts[COMET_FIXED_PROCESSOR_COUNT];
#endif
	static TaskContext* task_contexts;
	static uint64_t reseed_threshold;
	static uint32_t thread_stack_size;
	static uint32_t task_stack_size;
#if COMET_FIXED_PROCESSOR_COUNT == 0
	static uint32_t max_threads;
#else
	static constexpr uint32_t max_threads = COMET_FIXED_PROCESSOR_COUNT;
#endif
	static uint32_t max_tasks;
	static uint32_t queue_capacity;
	static uint8_t max_threads_log2;
	static uint8_t queue_capacity_log2;
	COMET_SHARED_ATTR static atomic<uint32_t> task_pool_bump;
	COMET_SHARED_ATTR static atomic<IndexPair> task_pool_flist;

	static atomic<SchedulerState> lib_state;

#ifdef COMET_DEBUG

	template <typename S>
	COMET_NOINLINE static void CustomAssertHandler(S&& text)
	{
		COMET_DEBUGTRAP;
		abort();
	}

#define COMET_SYMBOL_TO_STRING(E) #E
#include <cassert>
#define COMET_ASSERT(E, M) assert((E) && M)
#define COMET_INVARIANT(E) COMET_ASSERT((E), "Invariant broken: \"" COMET_SYMBOL_TO_STRING(E) "\"")
#else
#define COMET_ASSERT(E) (void)(E)
#define COMET_INVARIANT COMET_ASSUME
#endif

	namespace OS
	{
#ifdef __FreeBSD__
		COMET_INLINE static void* Malloc(size_t n)
		{
			auto page_size = getpagesize();
			auto r = mmap(nullptr, n + page_size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
			COMET_ASSERT(r != nullptr, "Failed to allocate memory.");
			auto v = mmap((uint8_t*)r + n, page_size, PROT_NONE, MAP_GUARD, -1, 0);
			COMET_ASSERT(v != MAP_FAILED, "Failed to set guard page.");
			return r;
		}

		COMET_INLINE static void Free(void* p, size_t n)
		{
			auto k = munmap(p, n);
			COMET_ASSERT(k == 0, "Failed to free memory.");
		}

		COMET_INLINE static ThreadHandle NewThread(ThreadEntryPoint fn, ThreadParam arg, uint32_t affinity)
		{
			pthread_t r;
			pthread_attr_t attr;
			pthread_attr_init(&attr);
			cpuset_t set = {};
			CPU_SET(affinity, &set);
			COMET_ASSERT(pthread_attr_setaffinity_np(&attr, sizeof(cpuset_t), &set) == 0, "Failed to set thread affinity.");
			COMET_ASSERT(pthread_attr_setstacksize(&attr, thread_stack_size) == 0, "Failed to set thread stack size.");
			COMET_ASSERT(pthread_create(&r, &attr, fn, arg) == 0, "Failed to create thread.");
			pthread_attr_destroy(&attr);
			return r;
		}

		COMET_INLINE static bool PauseThread(ThreadHandle thread)
		{
			return pthread_suspend_np(thread) == 0;
		}

		COMET_INLINE static bool ResumeThread(ThreadHandle thread)
		{
			return pthread_resume_np(thread) == 0;
		}

		COMET_INLINE static bool AwaitThread(ThreadHandle thread)
		{
			void* p;
			return pthread_join(thread, &p) == 0;
		}

		[[noreturn]]
		COMET_INLINE static void ExitThread()
		{
			pthread_exit(nullptr);
		}

#elif defined(_WIN32)
		COMET_INLINE static void* Malloc(size_t n)
		{
			//auto a = GetLargePageMinimum();
			//VirtualAlloc2()
			return VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		}

		COMET_INLINE static void Free(void* p, size_t n)
		{
			(void)VirtualFree(p, 0, MEM_RELEASE);
		}

		COMET_INLINE static ThreadHandle NewThread(ThreadEntryPoint fn, ThreadParam arg, uint32_t affinity)
		{
			ThreadHandle r = CreateThread(nullptr, thread_stack_size, fn, arg, CREATE_SUSPENDED, nullptr);
			GROUP_AFFINITY prior, desired = {};
			desired.Group = (WORD)(affinity >> 6);
			desired.Mask = (ULONG_PTR)1 << (affinity & 63);
			if (r != nullptr)
			{
				if (!SetThreadGroupAffinity(r, &desired, &prior))
				{
					(void)TerminateThread(r, MAXDWORD);
					return nullptr;
				}
				::ResumeThread(r);
			}
			return r;
		}

		COMET_INLINE static bool PauseThread(ThreadHandle thread)
		{
			return SuspendThread(thread) != MAXDWORD;
		}

		COMET_INLINE static bool ResumeThread(ThreadHandle thread)
		{
			return ::ResumeThread(thread) != MAXDWORD;
		}

		COMET_INLINE static bool AwaitThread(ThreadHandle thread)
		{
			return WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0;
		}

		[[noreturn]]
		COMET_INLINE static void ExitThread()
		{
			::ExitThread(0);
		}
#endif
	}

	template <SchedulerState From, SchedulerState To>
	COMET_INLINE bool SwitchState()
	{
		static_assert((uint8_t)From < (uint8_t)SchedulerState::MaxEnum);
		SchedulerState prior = From;
		if constexpr (From == SchedulerState::Uninitialized)
		{
			static_assert(To == SchedulerState::Initializing);
			return lib_state.compare_exchange_strong(prior, To, std::memory_order_acquire, std::memory_order_relaxed);
		}
		else if constexpr (From == SchedulerState::Initializing)
		{
			static_assert(To == SchedulerState::Running);
			lib_state.store(To, std::memory_order_release);
			return true;
		}
		else if constexpr (From == SchedulerState::Running)
		{
			static_assert(
				To == SchedulerState::Pausing ||
				To == SchedulerState::ShuttingDown);
			return lib_state.compare_exchange_strong(prior, To, std::memory_order_acquire, std::memory_order_relaxed);
		}
		else if constexpr (From == SchedulerState::Pausing)
		{
			static_assert(To == SchedulerState::Paused);
			lib_state.store(To, std::memory_order_release);
			return true;
		}
		else if constexpr (From == SchedulerState::Paused)
		{
			static_assert(To == SchedulerState::Resuming);
			return lib_state.compare_exchange_strong(prior, To, std::memory_order_acquire, std::memory_order_relaxed);
		}
		else if constexpr (From == SchedulerState::Resuming)
		{
			static_assert(To == SchedulerState::Running);
			lib_state.store(To, std::memory_order_release);
			return true;
		}
		else if constexpr (From == SchedulerState::ShuttingDown)
		{
			static_assert(To == SchedulerState::Finalizing);
			return lib_state.compare_exchange_strong(prior, To, std::memory_order_acquire, std::memory_order_relaxed);
		}
		else if constexpr (From == SchedulerState::Finalizing)
		{
			static_assert(To == SchedulerState::Uninitialized);
			lib_state.store(To, std::memory_order_release);
			return true;
		}
	}

	namespace Time
	{
		static uint64_t Get()
		{
#ifdef __FreeBSD__
			struct timespec spec;
			clock_gettime(CLOCK_REALTIME, &spec);
			return spec.tv_nsec;
#elif defined(_WIN32)
			LARGE_INTEGER k;
			QueryPerformanceCounter(&k);
			return k.QuadPart;
#endif
		}

		static uint64_t GetReseedThreshold(uint64_t ns)
		{
#ifdef _WIN32
			LARGE_INTEGER k;
			QueryPerformanceFrequency(&k);
			ns *= k.QuadPart;
#endif
			ns /= 1000000000;
			return ns;
		}
	}

	namespace RNG
	{
		static constexpr uint8_t POOL_COUNT = 32;
		static constexpr uint8_t POOL_MASK = POOL_COUNT - 1;

		struct COMET_SHARED_ATTR Pool
		{
			atomic<uint64_t> hash;
		};

		static Pool pools[POOL_COUNT];

		

		template <typename... T>
		COMET_FLATTEN static void AddEntropy(T... data)
		{
			auto index = (this_thread - thread_contexts) & POOL_MASK;
			(void)pools[index].hash.fetch_add(ReduceMix(data...), std::memory_order_relaxed);
		}

		COMET_NOINLINE static void ReseedThread(ThreadContext& here)
		{
			++here.reseed_counter;
			auto x = WellonsMix64(here.romu2jr[0] ^ here.romu2jr[1]);
#ifndef COMET_NO_RDRAND
			unsigned long long y = 0;
#ifdef __has_builtin
#if __has_builtin(__builtin_ia32_rdrand64_step)
			(void)__builtin_ia32_rdrand64_step(&y);
#elif defined(_MSVC_LANG)
			(void)_rdrand64_step(&y);
#endif
#elif defined(_MSVC_LANG)
			(void)_rdrand64_step(&y);
#endif
			x ^= (uint64_t)y;
#endif
			for (uint8_t i = 0; i != (std::countr_zero(here.reseed_counter) & POOL_MASK); ++i)
				x ^= NonAtomicRef(pools[i].hash);
			COMET_UNLIKELY_IF(x == 0)
				x = WellonsMix64(Time::Get());
			here.romu2jr[0] = x;
			here.romu2jr[1] = WellonsMix64(x);
			here.next_reseed = Time::Get() + reseed_threshold;
		}

		COMET_INLINE static uint32_t Get()
		{
			uint64_t x;
			COMET_UNLIKELY_IF(this_thread == nullptr)
				return WellonsMix64(Time::Get());
			auto& here = *this_thread;
			COMET_UNLIKELY_IF(Time::Get() >= here.next_reseed)
				ReseedThread(here);
			x = here.romu2jr[0];
			here.romu2jr[0] = here.romu2jr[1] * UINT64_C(15241094284759029579);
			here.romu2jr[1] = std::rotl(here.romu2jr[0] - x, 27);
			return (uint32_t)x;
		}

		COMET_INLINE static uint32_t GetRandomThreadIndex()
		{
			auto n = Get();
			if (std::popcount(max_threads) == 1)
				return (uint32_t)n & (max_threads - 1);
			n *= max_threads;
			n >>= (32 - max_threads_log2);
			COMET_INVARIANT(n < max_threads);
			return (uint32_t)n;
		}
	}

	COMET_INLINE static uint32_t AcquireTask()
	{
		for (;; COMET_SPIN)
		{
			auto prior = task_pool_flist.load(std::memory_order_acquire);
			COMET_UNLIKELY_IF(prior.first == UINT32_MAX)
				break;
			COMET_LIKELY_IF(task_pool_flist.compare_exchange_weak(prior, { task_contexts[prior.first].next, prior.second + 1 }, std::memory_order_acquire, std::memory_order_relaxed))
				return prior.first;
		}
		COMET_UNLIKELY_IF(NonAtomicRef(task_pool_bump) >= max_tasks)
			return UINT32_MAX;
		auto n = task_pool_bump.fetch_add(1, std::memory_order_acquire);
		COMET_UNLIKELY_IF(n >= max_tasks)
		{
			(void)task_pool_bump.fetch_sub(1, std::memory_order_release);
			return UINT32_MAX;
		}
		return n;
	}

	COMET_INLINE static void ReleaseTask(uint32_t index)
	{
		auto& task = task_contexts[index];
		for (;; COMET_SPIN)
		{
			auto prior = task_pool_flist.load(std::memory_order_acquire);
			task.next = prior.first;
			COMET_LIKELY_IF(task_pool_flist.compare_exchange_weak(prior, { index, prior.second + 1 }, std::memory_order_release, std::memory_order_relaxed))
				break;
		}
	}

	COMET_INLINE static uint32_t AdjustQueueIndex(uint32_t index)
	{
		return index & (queue_capacity - 1);
	}

	COMET_INLINE static uint32_t PopTask(ThreadContext& thread)
	{
		uint32_t m = 16;
		while (true)
		{
#if COMET_SLEEP_ON_IDLE
			thread.last_counter = thread.counter.load(std::memory_order_relaxed);
#endif
			uint8_t n = 0;
			do
			{
				COMET_UNLIKELY_IF(thread.quit_flag)
					return UINT32_MAX;
				for (auto& q : thread.queues)
				{
					if (q.size.load(std::memory_order_acquire) == 0)
						continue;
					auto& e = q.values[q.tail];
					COMET_UNLIKELY_IF(NonAtomicRef(e) == UINT32_MAX)
						continue;
					auto r = e.exchange(UINT32_MAX, std::memory_order_acquire);
					COMET_UNLIKELY_IF(r == UINT32_MAX)
						continue;
					++q.tail;
					q.tail = AdjustQueueIndex(q.tail);
					(void)q.size.fetch_sub(1, std::memory_order_release);
					return r;
				}
				++n;
			} while (n < m);
			m /= 2;
#if COMET_SLEEP_ON_IDLE
			thread.counter.wait(thread.last_counter, std::memory_order_acquire);
#endif
		}
	}

	COMET_INLINE static bool PushTask(ThreadContext& thread, TaskContext& task, uint32_t index)
	{
		auto& q = thread.queues[task.priority];
		auto n = NonAtomicRef(q.size);
		COMET_UNLIKELY_IF(n >= queue_capacity)
			return false;
		n = q.size.fetch_add(1, std::memory_order_acquire);
		COMET_UNLIKELY_IF(n >= queue_capacity)
		{
			(void)q.size.fetch_sub(1, std::memory_order_release);
			return false;
		}
		n = q.head.fetch_add(1, std::memory_order_acquire);
		n = AdjustQueueIndex(n);
		q.values[n].store(index, std::memory_order_release);
#if COMET_SLEEP_ON_IDLE
		(void)thread.counter.fetch_add(1, std::memory_order_release);
		thread.counter.notify_one();
#endif
		return true;
	}

	template <typename E>
	static void CreateContext(Context& c, E entry_point, void* parameter)
	{
		c = CreateFiberEx(task_stack_size, task_stack_size, FIBER_FLAG_FLOAT_SWITCH, entry_point, parameter);
	}

	static void DestroyContext(Context& c)
	{
		DeleteFiber(c);
		c = nullptr;
	}

	static void SwapContext(Context& from, Context& to)
	{
		SwitchToFiber(to);
	}

	static ThreadReturn COMET_THREAD_CALL ThreadMain(ThreadParam arg)
	{
		auto& here = thread_contexts[(uintptr_t)arg];
#ifdef _WIN32
		here.main_context = ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
#endif
		this_thread = &here;
		RNG::AddEntropy(Time::Get(), &here);
		while (true)
		{
			auto index = PopTask(here);
			COMET_UNLIKELY_IF(here.quit_flag)
				break;
			COMET_INVARIANT(index < max_tasks);
			here.this_task = task_contexts + index;
			SwapContext(here.main_context, here.this_task->context);
			++here.yield_count;
			if (here.this_task->fn != nullptr)
			{
				COMET_INVARIANT(here.this_task->sleeping != 2);
				if (here.this_task->sleeping == 1)
				{
					here.this_task->sleeping = 2;
					std::atomic_thread_fence(std::memory_order_release);
					continue;
				}
				auto target = this_thread;
				while (!PushTask(*target, *here.this_task, index))
					target = thread_contexts + RNG::GetRandomThreadIndex();
			}
			else
			{
				RNG::AddEntropy(here.local_accumulator, here.yield_count);
				if (here.this_task->counter != nullptr)
					here.this_task->counter->Decrement();
				ReleaseTask(index);
			}
		}
		return 0;
	}

	static void COMET_TASK_CALL TaskMain(void* arg)
	{
		while (true)
		{
			COMET_UNLIKELY_IF(this_thread->quit_flag)
				break;
			auto& task = *(TaskContext*)arg;
			COMET_INVARIANT(task.fn != nullptr);
			task.fn(task.arg);
			task.fn = nullptr;
			SwapContext(task.context, this_thread->main_context);
		}
		OS::ExitThread();
	}

	COMET_INLINE static void FinalizeInner()
	{
		for (uint32_t i = 0; i != NonAtomicRef(task_pool_bump); ++i)
			DestroyContext(task_contexts[i].context);
		size_t buffer_size =
			sizeof(ThreadContext) * max_threads +
			sizeof(TaskContext) * max_tasks;
		OS::Free(thread_contexts, buffer_size);
	}

	InitOptions InitOptions::Default()
	{
		InitOptions r = {};
		r.thread_stack_size = 1U << 21;
		r.task_stack_size = 1U << 16;
#if COMET_FIXED_PROCESSOR_COUNT == 0
#ifdef __FreeBSD__
		r.max_threads = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_WIN32)
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		r.max_threads = info.dwNumberOfProcessors;
#endif
		r.max_tasks = r.max_threads * 256;
#else
		r.max_tasks = COMET_FIXED_PROCESSOR_COUNT * 256;
#endif
		r.reseed_threshold_ns = UINT32_MAX; // ~4s
		return r;
	}

	bool Init(const InitOptions& options)
	{
		COMET_UNLIKELY_IF(!SwitchState<SchedulerState::Uninitialized, SchedulerState::Initializing>())
			return false;

		COMET_INVARIANT(options.max_tasks != 0);
#if COMET_FIXED_PROCESSOR_COUNT == 0
		COMET_INVARIANT(options.max_threads != 0);
#endif
		COMET_INVARIANT(options.reseed_threshold_ns != 0);
		COMET_INVARIANT(options.task_stack_size != 0);
		COMET_INVARIANT(options.thread_stack_size != 0);
#if COMET_FIXED_PROCESSOR_COUNT == 0
		max_threads = options.max_threads;
#endif
		max_tasks = options.max_tasks;
		queue_capacity = 1U << (31 - std::countl_zero((max_tasks / max_threads) - 1));
		queue_capacity_log2 = 31 - std::countl_zero(queue_capacity);
		max_threads_log2 = 31 - std::countl_zero(max_threads);
#if COMET_FIXED_PROCESSOR_COUNT == 0
		size_t buffer_size =
			sizeof(ThreadContext) * max_threads +
			sizeof(atomic<uint32_t>) * queue_capacity * max_threads * COMET_MAX_PRIORITY +
			sizeof(TaskContext) * max_tasks;
		thread_contexts = (ThreadContext*)OS::Malloc(buffer_size);
		task_contexts = (TaskContext*)(thread_contexts + max_threads);
#else
		size_t buffer_size =
			sizeof(atomic<uint32_t>) * queue_capacity * max_threads * MAX_PRIORITY +
			sizeof(TaskContext) * max_tasks;
		task_contexts = (TaskContext*)OS::Malloc(buffer_size);
#endif
		task_stack_size = options.task_stack_size;
		thread_stack_size = options.thread_stack_size;
		uint8_t* bump = (uint8_t*)(task_contexts + max_tasks);
		auto qn = sizeof(atomic<uint32_t>) * queue_capacity;
		for (uint32_t i = 0; i != max_threads; ++i)
		{
			auto& e = thread_contexts[i];
			e.handle = OS::NewThread(ThreadMain, (ThreadParam)(uintptr_t)i, i);
			for (auto& q : e.queues)
			{
				q.values = (atomic<uint32_t>*)bump;
				(void)memset(bump, 0xff, qn);
				bump += qn;
			}
		}
		NonAtomicRef(task_pool_flist) = { UINT32_MAX, 0 };
		NonAtomicRef(task_pool_bump) = 0;
		reseed_threshold = Time::GetReseedThreshold(options.reseed_threshold_ns);
		return SwitchState<SchedulerState::Initializing, SchedulerState::Running>();
	}

	bool Pause()
	{
		COMET_UNLIKELY_IF(!SwitchState<SchedulerState::Running, SchedulerState::Pausing>())
			return false;
		for (uint32_t i = 0; i != max_threads; ++i)
			COMET_UNLIKELY_IF(!OS::PauseThread(thread_contexts[i].handle))
				return false;
		return SwitchState<SchedulerState::Pausing, SchedulerState::Paused>();
	}

	bool Resume()
	{
		COMET_UNLIKELY_IF(!SwitchState<SchedulerState::Paused, SchedulerState::Resuming>())
			return false;
		for (uint32_t i = 0; i != max_threads; ++i)
			COMET_UNLIKELY_IF(!OS::ResumeThread(thread_contexts[i].handle))
				return false;
		return SwitchState<SchedulerState::Resuming, SchedulerState::Running>();
	}

	bool Shutdown()
	{
		COMET_UNLIKELY_IF(!SwitchState<SchedulerState::Running, SchedulerState::ShuttingDown>())
			return false;
		for (uint32_t i = 0; i != max_threads; ++i)
		{
			auto& thread = thread_contexts[i];
			thread.quit_flag = true;
#if COMET_SLEEP_ON_IDLE
			(void)thread.counter.fetch_add(1, std::memory_order_release);
			thread.counter.notify_one();
#endif
		}
		return true;
	}

	bool TryFinalize()
	{
		COMET_UNLIKELY_IF(!SwitchState<SchedulerState::ShuttingDown, SchedulerState::Finalizing>())
			return false;
		return Finalize();
	}

	bool Finalize()
	{
		for (uint32_t i = 0; i != max_threads; ++i)
		{
			auto& thread = thread_contexts[i];
#if COMET_SLEEP_ON_IDLE
			(void)thread.counter.fetch_add(1, std::memory_order_release);
			thread.counter.notify_one();
#endif
			COMET_UNLIKELY_IF(!OS::AwaitThread(thread.handle))
				return false;
		}
		FinalizeInner();
		return SwitchState<SchedulerState::Finalizing, SchedulerState::Uninitialized>();
	}

	SchedulerState GetSchedulerState()
	{
		return lib_state.load(std::memory_order_acquire);
	}

	bool IsTask()
	{
		return this_thread != nullptr;
	}

	DispatchResult Dispatch(void(*fn)(void* arg), void* arg)
	{
		return Dispatch(fn, arg, {});
	}

	DispatchResult Dispatch(void(*fn)(void* arg), void* arg, TaskOptions options)
	{
		auto index = AcquireTask();
		COMET_UNLIKELY_IF(index == UINT32_MAX)
		{
			switch (options.error_policy)
			{
			case DispatchErrorPolicy::RunSequentially:
				fn(arg);
				if (options.counter != nullptr)
					options.counter->Decrement();
				return DispatchResult::Sequential;
			case DispatchErrorPolicy::Spin:
				for (;; COMET_SPIN)
				{
					index = AcquireTask();
					COMET_UNLIKELY_IF(index == UINT32_MAX)
						break;
				}
				break;
			case DispatchErrorPolicy::Return:
				return DispatchResult::Failure;
			default:
				COMET_UNREACHABLE;
			}
		}
		auto& task = task_contexts[index];
		COMET_UNLIKELY_IF(!IsValidContext(task.context))
			CreateContext(task.context, TaskMain, &task);
		task.fn = fn;
		task.arg = arg;
		task.priority = options.priority;
		task.counter = options.counter;
		task.next = UINT32_MAX;
		if (options.preferred_thread != nullptr)
		{
			COMET_UNLIKELY_IF(!PushTask(thread_contexts[*options.preferred_thread], task, index))
			{
				switch (options.error_policy)
				{
				case DispatchErrorPolicy::RunSequentially:
					fn(arg);
					if (options.counter != nullptr)
						options.counter->Decrement();
					return DispatchResult::Sequential;
				case DispatchErrorPolicy::Spin:
					while (!PushTask(thread_contexts[*options.preferred_thread], task, index))
						COMET_SPIN;
					break;
				case DispatchErrorPolicy::Return:
					return DispatchResult::Failure;
				default:
					COMET_UNREACHABLE;
				}
			}
		}
		else
		{
			auto target = this_thread;
			if (target == nullptr)
				target = thread_contexts + RNG::GetRandomThreadIndex();
			if (!PushTask(*target, task, index))
			{
				switch (options.error_policy)
				{
				case DispatchErrorPolicy::RunSequentially:
					fn(arg);
					if (options.counter != nullptr)
						options.counter->Decrement();
					return DispatchResult::Sequential;
				case DispatchErrorPolicy::Spin:
					while (!PushTask(thread_contexts[RNG::GetRandomThreadIndex()], task, index))
						COMET_SPIN;
					break;
				case DispatchErrorPolicy::Return:
					return DispatchResult::Failure;
				default:
					COMET_UNREACHABLE;
				}
			}
		}
		return DispatchResult::Success;
	}

	void Yield()
	{
#ifdef COMET_DEBUG
		COMET_ASSERT(!this_thread->yield_trap, "Attempted to invoke Yield() while yield_trap was set to true.");
#endif
#ifdef _MSVC_LANG
		auto mask = (uintptr_t)_ReturnAddress();
#elif defined(__clang__)
		auto mask = (uintptr_t)__builtin_return_address(0);
#endif
		this_thread->local_accumulator ^= ReduceMix(mask);
		SwapContext(this_thread->this_task->context, this_thread->main_context);
	}

	void Exit()
	{
		this_thread->this_task->fn = nullptr;
		Yield();
	}

	uint64_t ThisTaskID()
	{
		auto task = this_thread->this_task;
		return ((uint64_t)task->generation << 32) | (task - task_contexts);
	}

	uint32_t WorkerThreadIndex()
	{
		return (uint32_t)(this_thread - thread_contexts);
	}

	uint32_t MaxTasks()
	{
		return max_tasks;
	}

#if COMET_FIXED_PROCESSOR_COUNT == 0
	uint32_t WorkerThreadCount()
	{
		return max_threads;
	}
#endif

	void Fence::Signal()
	{
		auto& self = *(atomic<uint32_t>*)this;
		auto index = self.load(std::memory_order_acquire);
		if (index == UINT32_MAX)
			return;
		auto& task = task_contexts[index];
		while (task.sleeping != 2)
			COMET_SPIN;
		task.sleeping = 0;
		auto target = this_thread;
		while (!PushTask(*target, task, index))
			target = thread_contexts + RNG::GetRandomThreadIndex();
	}

	void Fence::Await()
	{
		COMET_ASSERT(IsTask(), "Attempted to wait for a Fence from outside a Comet thread.");
		auto& self = *(atomic<uint32_t>*)this;
		auto index = (uint32_t)(this_thread->this_task - task_contexts);
		NonAtomicRef(self) = index;
		this_thread->this_task->sleeping = 1;
		Yield();
	}
	
	void Fence::Reset()
	{
		state = UINT32_MAX;
	}

	bool Counter::Decrement()
	{
		auto& self = *(CounterState*)this;
		COMET_LIKELY_IF(self.counter.fetch_sub(1, std::memory_order_acquire) != 1)
			return false;
		return WakeAll();
	}

	bool Counter::WakeAll()
	{
		auto& self = *(CounterState*)this;
		auto prior = self.queue.exchange({ 0, UINT32_MAX }, std::memory_order_acquire);
		if (prior.first == UINT32_MAX)
			return true;
		if (prior.second == UINT32_MAX)
			return false;
		auto i = prior.first;
		while (true)
		{
			auto& task = task_contexts[i];
			std::atomic_thread_fence(std::memory_order_acquire);
			while (task.sleeping != 2)
				COMET_SPIN;
			auto next = task.next;
			task.next = UINT32_MAX;
			task.sleeping = 0;
			auto target = this_thread;
			while (!PushTask(*target, task, i))
				target = thread_contexts + RNG::GetRandomThreadIndex();
			if (i == prior.second)
				return true;
			i = next;
		}
	}

	bool Counter::Await()
	{
		auto& self = *(CounterState*)this;
		IndexPair prior;
		auto index = (uint32_t)(this_thread->this_task - task_contexts);
		for (;; COMET_SPIN)
		{
			prior = self.queue.load(std::memory_order_acquire);
			if (prior.first != UINT32_MAX && prior.second == UINT32_MAX)
				return false;
			if (self.queue.compare_exchange_weak(prior, { prior.first == UINT32_MAX ? index : prior.first, index }, std::memory_order_acquire, std::memory_order_relaxed))
				break;
		}
		if (prior.second != UINT32_MAX)
		{
			COMET_INVARIANT(task_contexts[prior.first].next == UINT32_MAX);
			task_contexts[prior.second].next = index;
		}
		this_thread->this_task->sleeping = 1;
		Yield();
		return true;
	}

	void Counter::Reset(uint64_t new_value)
	{
		WakeAll();
		state[0] = new_value;
		state[1] = UINT64_MAX;
	}

	uint64_t Counter::Value() const
	{
		auto& self = *(CounterState*)this;
		return self.counter.load(std::memory_order_relaxed);
	}

	void MutexMCS::Lock(Node& node)
	{
		auto& self = *(atomic<Node*>*)this;
		node.index = (uint32_t)(this_thread->this_task - task_contexts);
		auto prior = self.exchange(&node, std::memory_order_acquire);
		COMET_LIKELY_IF(prior == nullptr)
			return;
		prior->next = &node;
		this_thread->this_task->sleeping = 1;
		Yield();
	}

	bool MutexMCS::IsLocked() const
	{
		return state != 0;
	}

	bool MutexMCS::TryLock(Node& node)
	{
		auto& self = *(atomic<Node*>*)this;
		node.index = (uint32_t)(this_thread->this_task - task_contexts);
		Node* expected = nullptr;
		return self.compare_exchange_weak(expected, &node, std::memory_order_acquire, std::memory_order_relaxed);
	}

	void MutexMCS::Unlock(Node& node)
	{
		auto& self = *(atomic<Node*>*)this;
		Node* expected = &node;
		COMET_LIKELY_IF(self.compare_exchange_strong(expected, nullptr, std::memory_order_release, std::memory_order_relaxed))
			return;
		auto& task = task_contexts[node.next->index];
		std::atomic_thread_fence(std::memory_order_acquire);
		while (task.sleeping != 2)
			COMET_SPIN;
		task.sleeping = 0;
		auto target = this_thread;
		while (!PushTask(*target, task, node.next->index))
			target = thread_contexts + RNG::GetRandomThreadIndex();
	}

	namespace Debug
	{
#ifdef COMET_DEBUG
		void SetYieldTrap(bool value)
		{
			this_thread->yield_trap = value;
		}
#endif
	}

	namespace RCU
	{
		uint32_t GetSchedulerSnapshotSize()
		{
			return (max_threads - 1) * sizeof(ThreadContext::yield_count);
		}

		void GetSchedulerSnapshot(void* out)
		{
			auto ptr = (uint32_t*)out;
			auto here_index = (uint32_t)(this_thread - thread_contexts);
			for (uint32_t i = 0; i != max_threads; ++i)
			{
				COMET_UNLIKELY_IF(i == here_index)
					continue;
				std::atomic_thread_fence(std::memory_order_acquire);
				*ptr = thread_contexts[i].yield_count;
				++ptr;
			}
		}

		uint32_t TrySync(void* snapshot, uint32_t prior_result)
		{
			COMET_ASSERT(prior_result < max_threads, "Invalid prior_result value passed to Comet::RCU::TrySync.");
			auto ptr = (uint32_t*)snapshot;
			auto i = prior_result;
			for (; i != max_threads; ++i)
			{
				std::atomic_thread_fence(std::memory_order_acquire);
				auto current = thread_contexts[i].yield_count;
				auto prior = *ptr;
				COMET_UNLIKELY_IF(current == prior)
					break;
			}
			return i;
		}
	}
}
#endif