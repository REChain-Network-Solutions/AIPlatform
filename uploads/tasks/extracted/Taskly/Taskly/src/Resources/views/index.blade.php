@extends('layouts.main')

@section('page-title')
    {{ __('Dashboard') }}
@endsection

@section('page-breadcrumb')
    {{ __('Project')}}
@endsection

@section('content')
    <div class="row row-gap mb-4 ">
        <div class="col-xl-6 col-12">
            <div class="dashboard-card">
                <img src="{{ asset('assets/images/layer.png')}}" class="dashboard-card-layer" alt="layer">
                <div class="card-inner">
                    <div class="card-content">
                        <h2>{{Auth::user()->ActiveWorkspaceName()}}</h2>
                        <p>{{__('Optimizes project management with task tracking, timelines, and real-time progress updates.') }}</p>
                    </div>
                    <div class="card-icon  d-flex align-items-center justify-content-center">
                        <svg xmlns="http://www.w3.org/2000/svg" width="115" height="114" viewBox="0 0 115 114" fill="none">
                            <path d="M40.1631 26.8857C31.7824 36.4796 26.017 48.5014 23.2829 62.6554L3.54956 66.564C3.37125 66.6232 3.13346 66.6233 2.95515 66.6233C2.06359 66.6233 1.23143 66.268 0.696494 65.5573C-0.0167561 64.7282 -0.194989 63.5437 0.221074 62.537L5.92702 48.3832C10.0876 37.901 19.6572 30.202 30.8908 28.4254L40.1631 26.8857Z" fill="#18BF6B"/>
                            <path d="M87.416 73.9658L85.8707 83.2043C84.0876 94.3972 76.3607 103.932 65.7808 108.137L51.6347 113.763C51.278 113.94 50.862 114 50.5053 114C49.8515 114 49.1382 113.763 48.6033 113.289C47.7712 112.638 47.3552 111.512 47.593 110.446L51.5159 90.7847C65.7214 88.0605 77.7871 82.316 87.416 73.9658Z" fill="#18BF6B"/>
                            <path opacity="0.7" d="M109.943 0C61.8106 0 30.1244 23.4516 23.0098 64.344C22.2787 68.5132 23.6934 72.8424 26.7841 75.9219L38.208 87.3102C40.7638 89.8508 44.1756 91.2543 47.6467 91.2543C48.3719 91.2543 49.1028 91.195 49.8279 91.0707C90.8695 83.9759 114.407 52.4051 114.407 4.45354C114.401 1.98992 112.404 0 109.943 0ZM48.3004 82.306C46.9869 82.5251 45.5307 82.0394 44.5083 81.0208L33.0845 69.6325C32.0562 68.6139 31.5629 67.163 31.7888 65.8602C36.6032 38.2097 52.9127 23.8603 70.1437 16.4695C85.0625 17.3874 96.95 29.2259 97.8713 44.0964C90.4535 61.2647 76.0578 77.5091 48.3004 82.306Z" fill="#18BF6B"/>
                            <path d="M65.3646 62.1809C72.7506 62.1809 78.7381 56.2152 78.7381 48.8561C78.7381 41.497 72.7506 35.5312 65.3646 35.5312C57.9787 35.5312 51.9912 41.497 51.9912 48.8561C51.9912 56.2152 57.9787 62.1809 65.3646 62.1809Z" fill="#18BF6B"/>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-xl-6 col-12">
            <div class="row dashboard-wrp">
                <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="fas fa-tasks text-danger"></i>
                                </div>
                                <a href="{{ route('projects.index') }}"><h3 class="mt-3 mb-0 text-danger">{{ __('Total Project') }}</h3></a>
                            </div>
                            <h3 class="mb-0">{{ $totalProject }}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="ti ti-file-invoice"></i>
                                </div>
                            <h3 class="mt-3 mb-0">{{ __('Total Task') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $totalTask }}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="fas fa-bug"></i>
                                </div>
                                <h3 class="mt-3 mb-0">{{ __('Total Bug') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $totalBugs }}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="fas fa-users"></i>
                                </div>
                                <h3 class="mt-3 mb-0">{{ __('Total User') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $totalMembers }}</h3>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-xxl-7 d-flex flex-column">

            <div class="card h-100">
                <div class="card-header">
                    <div>
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h5 class="mb-0">
                                    {{ __('Tasks') }}
                                </h5>
                            </div>
                            <div class="float-end">
                                <small><b>{{ $completeTask }}</b> {{ __('Tasks completed out of') }}
                                    {{ $totaltasks }}</small>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card-body ">
                    <div class="table-responsive custom-scrollbar" style="max-height: 515px; overflow-y:auto;" >
                        <table class="table table-centered table-hover mb-0 animated">
                            <tbody>
                                @forelse($tasks as $task)
                                    <tr>
                                        <td>
                                            <div class="font-14 my-1"><a
                                                    href="{{ route('projects.task.board', [$task->project_id]) }}"
                                                    class="text-body">{{ $task->title }}</a></div>

                                            @php($due_date = '<span class="text-' . ($task->due_date < date('Y-m-d') ? 'danger' : 'success') . '">' . date('Y-m-d', strtotime($task->due_date)) . '</span> ')

                                            <span class="text-muted font-13">{{ __('Due Date') }} :
                                                {!! $due_date !!}</span>
                                        </td>
                                        <td>
                                            <span class="text-muted font-13">{{ __('Status') }}</span> <br />
                                            @if ($task->complete == '1')
                                                <span
                                                    class="badge bg-success p-2 px-3">{{ __($task->status) }}</span>
                                            @else
                                                <span
                                                    class="badge bg-primary p-2 px-3">{{ __($task->status) }}</span>
                                            @endif
                                        </td>
                                        <td>
                                            <span class="text-muted font-13">{{ __('Project') }}</span>
                                            <div class="font-14 mt-1 font-weight-normal">{{ $task->project->name }}</div>
                                        </td>
                                        @if (Auth::user()->hasRole('client') || Auth::user()->hasRole('client'))
                                            <td>
                                                <span class="text-muted font-13">{{ __('Assigned to') }}</span>
                                                <div class="font-14 mt-1 font-weight-normal">
                                                    @foreach ($task->users() as $user)
                                                        <span
                                                            class="badge p-2 px-2 bg-secondary">{{ isset($user->name) ? $user->name : '-' }}</span>
                                                    @endforeach
                                                </div>
                                            </td>
                                        @endif
                                    </tr>
                                @empty
                                    @include('layouts.nodatafound')
                                @endforelse
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-xxl-5">
            <div class="card">
                <div class="card-header">
                    <h5>{{ __('Tasks Overview') }}</h5>
                </div>
                <div class="card-body p-2">
                    <div id="task-area-chart"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div class="float-end">
                        <a href="#" data-bs-toggle="tooltip" data-bs-placement="top" title="Refferals"><i
                                class=""></i></a>
                    </div>
                    <h5>{{ __('Project Status') }}</h5>
                </div>
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-sm-8">
                            <div id="projects-chart"></div>
                        </div>
                        <div class="col-sm-4">
                            <div class="col-6">
                                <span class="d-flex align-items-center mb-2">
                                    <i class="f-10 lh-1 fas fa-circle text-danger"></i>
                                    <span class="ms-2 text-sm">{{ __('On Going') }}</span>
                                </span>
                            </div>
                            <div class="col-6">
                                <span class="d-flex align-items-center mb-2">
                                    <i class="f-10 lh-1 fas fa-circle text-warning"></i>
                                    <span class="ms-2 text-sm">{{ __('On Hold') }}</span>
                                </span>
                            </div>
                            <div class="col-6">
                                <span class="d-flex align-items-center mb-2">
                                    <i class="f-10 lh-1 fas fa-circle text-primary"></i>
                                    <span class="ms-2 text-sm">{{ __('Finished') }}</span>
                                </span>
                            </div>
                        </div>
                        <div class="row text-center">
                            @foreach ($arrProcessPer as $index => $value)
                                <div class="col-4">
                                    <i class="fas fa-chart"></i>
                                    <h6 class="font-weight-bold">
                                        <span>{{ $value }}%</span>
                                    </h6>
                                    <p class="text-muted">{{ __($arrProcessLabel[$index]) }}</p>
                                </div>
                            @endforeach
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection

@push('scripts')
    <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>
    <script>
        (function() {
            var options = {
                chart: {
                    height: 170,
                    type: 'donut',
                },
                dataLabels: {
                    enabled: false,
                },
                plotOptions: {
                    pie: {
                        donut: {
                            size: '70%',
                        }
                    }
                },


                series: {!! json_encode($arrProcessPer) !!},
                colors: ['#FF3A6E', '#6fd943', '#ffa21d'],
                labels: {!! json_encode($arrProcessLabel) !!},
                grid: {
                    borderColor: '#e7e7e7',
                    row: {
                        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
                        opacity: 0.5
                    },
                },
                markers: {
                    size: 1
                },
                legend: {
                    show: false
                }
            };
            var chart = new ApexCharts(document.querySelector("#projects-chart"), options);
            chart.render();

        })();
    </script>
    <script>
        (function() {
            var options = {
                chart: {
                    height: 157,
                    type: 'line',
                    toolbar: {
                        show: false,
                    },
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    width: 2,
                    curve: 'smooth'
                },
                series: [
                    @foreach ($chartData['stages'] as $id => $name)
                        {
                            name: "{{ __($name) }}",
                            data: {!! json_encode($chartData[$id]) !!}
                        },
                    @endforeach
                ],
                xaxis: {
                    categories: {!! json_encode($chartData['label']) !!},
                },
                colors: {!! json_encode($chartData['color']) !!},

                grid: {
                    strokeDashArray: 4,
                },
                legend: {
                    show: false,
                },
                yaxis: {
                    tickAmount: 5,
                    min: 1,
                    max: 40,
                },
            };
            var chart = new ApexCharts(document.querySelector("#task-area-chart"), options);
            chart.render();
        })();
    </script>
@endpush
