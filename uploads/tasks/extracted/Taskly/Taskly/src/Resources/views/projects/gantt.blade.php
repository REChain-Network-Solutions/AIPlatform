@extends('layouts.main')


@section('page-title') {{__('Gantt Chart')}} @endsection
@section('page-breadcrumb')
{{ __('Project Details') }},{{ __("Gantt Chart") }}
@endsection
@section('page-action')
    <div class="d-flex flex-wrap gap-2">
        <div class="btn-group flex-wrap bg-primary total-card" id="change_view" role="group">
            <a href="{{route('projects.gantt',[$project->id,'Quarter Day'])}}" class="btn btn-sm px-3 btn-primary @if($duration == 'Quarter Day')active @endif" data-value="Quarter Day">{{__('Quarter Day')}}</a>
            <a href="{{route('projects.gantt',[$project->id,'Half Day'])}}" class="btn btn-sm  px-3 btn-primary @if($duration == 'Half Day')active @endif" data-value="Half Day">{{__('Half Day')}}</a>
            <a href="{{route('projects.gantt',[$project->id,'Day'])}}" class="btn btn-sm px-3 btn-primary @if($duration == 'Day')active @endif" data-value="Day">{{__('Day')}}</a>
            <a href="{{route('projects.gantt',[$project->id,'Week'])}}" class="btn btn-sm px-3 btn-primary @if($duration == 'Week')active @endif" data-value="Week">{{__('Week')}}</a>
            <a href="{{route('projects.gantt',[$project->id,'Month'])}}" class="btn btn-sm px-3 btn-primary @if($duration == 'Month')active @endif" data-value="Month">{{__('Month')}}</a>
        </div>
        <div class="col-auto d-inline">
            <a href="{{route('projects.show',[$project->id])}}" class="btn btn-sm btn-primary" data-toggle="tooltip" title="{{__('Back')}}">
                <i class=" ti ti-arrow-back-up"></i>
            </a>
        </div>
    </div>
@endsection

@section('content')
    <section class="section mt-4">
        @if($project)
            <div class="row">
                <div class="col-12">
                    <div class="gantt-target"></div>
                </div>
            </div>
        @else
            <div class="container mt-5">
                <div class="card">
                    <div class="card-body p-4">
                        <div class="page-error">
                            <div class="page-inner">
                                <h1>404</h1>
                                <div class="page-description">
                                    {{ __('Page Not Found') }}
                                </div>
                                <div class="page-search">
                                    <p class="text-muted mt-3">{{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.")}}</p>
                                    <div class="mt-3">
                                        <a class="btn-return-home badge-blue" href="{{route('home')}}"><i class="fas fa-reply"></i> {{ __('Return Home')}}</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        @endif
    </section>
@endsection
@if($project)
    @push('css')
        <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/css/frappe-gantt.css')}}" />
    @endpush
    @push('scripts')
        @php
            $currantLang = basename(App::getLocale());
        @endphp
        <script>
            const month_names = {
                "{{$currantLang}}": [
                    '{{__('January')}}',
                    '{{__('February')}}',
                    '{{__('March')}}',
                    '{{__('April')}}',
                    '{{__('May')}}',
                    '{{__('June')}}',
                    '{{__('July')}}',
                    '{{__('August')}}',
                    '{{__('September')}}',
                    '{{__('October')}}',
                    '{{__('November')}}',
                    '{{__('December')}}'
                ],
                "en": [
                    'January',
                    'February',
                    'March',
                    'April',
                    'May',
                    'June',
                    'July',
                    'August',
                    'September',
                    'October',
                    'November',
                    'December'
                ],
            };
        </script>
        <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/frappe-gantt.js')}}"></script>
        <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/libs/moment/min/moment.min.js')}}"></script>
        <script>
            var tasks = JSON.parse('{!! addslashes(json_encode($tasks)) !!}');
            if(tasks.length != 0)
            {
                var gantt_chart = new Gantt(".gantt-target", tasks, {
                    custom_popup_html: function(task) {
                        var status_class = 'success';
                        if(task.custom_class == 'medium'){
                            status_class = 'info'
                        }else if(task.custom_class == 'high'){
                            status_class = 'danger'
                        }
                        return `<div class="details-container">
                                    <div class="title">${task.name} <span class="bg bg-${status_class} float-right">${task.extra.priority}</span></div>
                                    <div class="subtitle">
                                        <b>${task.progress}%</b> {{ __('Progress')}} <br>
                                        <b>${task.extra.comments}</b> {{ __('Comments')}} <br>
                                        <b>{{ __('Duration')}}</b> ${task.extra.duration}
                                    </div>
                                </div>
                              `;
                    },
                    on_click: function (task) {
                    },
                    on_date_change: function(task, start, end) {
                        task_id = task.id;
                        start = moment(start);
                        end = moment(end);
                        $.ajax({
                            url:"{{route('projects.gantt.post',[$project->id])}}",
                            data:{
                                start:start.format('YYYY-MM-DD HH:mm:ss'),
                                end:end.format('YYYY-MM-DD HH:mm:ss'),
                                task_id:task_id,
                            },
                            type:'POST',
                            success:function (data) {

                            },
                            error:function (data) {
                                toastrs('Error', '{{ __("Some Thing Is Wrong!")}}', 'error');
                            }
                        });
                    },
                    view_mode: '{{$duration}}',
                    language: '{{$currantLang}}'
                });
            }
            else
            {
                $('.gantt-target').append(`<div class="col-12">
                    <div class="card ">
                        <div class="card-body table-border-style mt-4">
                            <div class="table-responsive">
                                <table class="table ">
                                    <tr>
                                        <td colspan="10" class="text-center">
                                            <div class="text-center">
                                                <i class="fas fa-folder-open text-primary fs-40"></i>
                                                <h2>{{ __('Opps...') }}</h2>
                                                <h6> {!! __('No Data Found') !!} </h6>
                                                <small>{{__("Please create at least one task to show here.. ")}}
                                            </div>
                                        </td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>`);
            }
        </script>
    @endpush
@endif
