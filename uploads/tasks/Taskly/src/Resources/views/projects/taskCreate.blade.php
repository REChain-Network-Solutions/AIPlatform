@if($project && $currentWorkspace)
    <form class="needs-validation" method="post" action="{{ route('tasks.store',[$project->id]) }}" novalidate>
        @csrf
        <div class="modal-body">
            <div class="text-end">
                @if (module_is_active('AIAssistant'))
                    @include('aiassistant::ai.generate_ai_btn',['template_module' => 'project task','module'=>'Taskly'])
                @endif
            </div>
            <div class="row">
                <div class="form-group col-md-8">
                    <label class="form-label">{{ __('Project')}}</label><x-required></x-required>
                    <select class="form-control form-control-light select2" name="project_id" required>

                            <option value="{{$project->id}}">{{$project->name}}</option>

                    </select>
                </div>
                <div class="form-group col-md-4">
                    <label class="form-label">{{ __('Milestone')}}</label>
                    <select class="form-control form-control-light select2" name="milestone_id" id="task-milestone">
                        <option value="">{{__('Select Milestone')}}</option>
                        @foreach($project->milestones as $milestone)
                            <option value="{{$milestone->id}}">{{$milestone->title}}</option>
                        @endforeach
                    </select>
                </div>
                <div class="form-group col-md-8">
                    <label class="form-label">{{ __('Title')}}</label><x-required></x-required>
                    <input type="text" class="form-control form-control-light" id="task-title" placeholder="{{ __('Enter Title')}}" name="title" required>
                </div>
                <div class="form-group col-md-4">
                    <label class="form-label">{{ __('Priority')}}</label>
                    <select class="form-control form-control-light" name="priority" id="task-priority" required>
                        <option value="Low">{{ __('Low')}}</option>
                        <option value="Medium">{{ __('Medium')}}</option>
                        <option value="High">{{ __('High')}}</option>
                    </select>
                </div>
                <div class="form-group col-md-12">
                    <label class="form-label">{{ __('Assign To')}}</label><x-required></x-required>

                    <select class=" multi-select choices" id="assign_to" name="assign_to[]"  multiple="multiple" data-placeholder="{{ __('Select Users ...') }}" required>
                        @foreach($users as $u)
                            <option value="{{$u->id}}">{{$u->name}} - {{$u->email}}</option>
                        @endforeach
                    </select>
                    <p class="text-danger d-none" id="user_validation">{{__('Assign To filed is required.')}}</p>
                </div>

                <div class="form-group col-md-12">
                    <label class="form-label">{{ __('Duration')}}</label><x-required></x-required>
                    <div class='input-group'>
                        <input type='text' class=" form-control form-control-light" id="duration" name="duration" required autocomplete="off"
                            placeholder="Select date range" />
                            <input type="hidden" name="start_date">
                                <input type="hidden" name="due_date">
                                <span class="input-group-text"><i class="feather icon-calendar"></i></span>
                    </div>
                </div>
                <div class="form-group col-md-12">
                    <label class="form-label">{{ __('Description')}}</label><x-required></x-required>
                    <textarea class="form-control form-control-light" id="task-description" rows="3" name="description" placeholder="Enter Description" required></textarea>
                </div>
            </div>
            @if(module_is_active('CustomField') && !$customFields->isEmpty())
                <div class="col-md-12">
                    <div class="tab-pane fade show" id="tab-2" role="tabpanel">
                        @include('custom-field::formBuilder')
                    </div>
                </div>
            @endif
            @stack('calendar')
        </div>
        <div class="modal-footer">
            <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
            <input type="submit" value="{{ __('Create')}}" class="btn  btn-primary" id="submit">
        </div>
    </form>
     <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/libs/bootstrap-daterangepicker/daterangepicker.css')}} ">
     <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/libs/moment/min/moment.min.js')}}"></script>
     <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/libs/bootstrap-daterangepicker/daterangepicker.js')}}"></script>

    <script>
        $(function () {
            var start = moment('{{ date('Y-m-d') }}', 'YYYY-MM-DD HH:mm:ss');
            var end = moment('{{ date('Y-m-d') }}', 'YYYY-MM-DD HH:mm:ss');

            function cb(start, end) {
                $("form #duration").val(start.format('MMM D, YY hh:mm A') + ' - ' + end.format('MMM D, YY hh:mm A'));
                $('form input[name="start_date"]').val(start.format('YYYY-MM-DD HH:mm:ss'));
                $('form input[name="due_date"]').val(end.format('YYYY-MM-DD HH:mm:ss'));
            }

            $('form #duration').daterangepicker({
                autoApply: true,
                timePicker: true,
                autoUpdateInput: false,
                startDate: start,
                endDate: end,
                locale: {
                    format: 'MMMM D, YYYY hh:mm A',
                    applyLabel: "{{__('Apply')}}",
                    cancelLabel: "{{__('Cancel')}}",
                    fromLabel: "{{__('From')}}",
                    toLabel: "{{__('To')}}",
                    daysOfWeek: [
                        "{{__('Sun')}}",
                        "{{__('Mon')}}",
                        "{{__('Tue')}}",
                        "{{__('Wed')}}",
                        "{{__('Thu')}}",
                        "{{__('Fri')}}",
                        "{{__('Sat')}}"
                    ],
                    monthNames: [
                        "{{__('January')}}",
                        "{{__('February')}}",
                        "{{__('March')}}",
                        "{{__('April')}}",
                        "{{__('May')}}",
                        "{{__('June')}}",
                        "{{__('July')}}",
                        "{{__('August')}}",
                        "{{__('September')}}",
                        "{{__('October')}}",
                        "{{__('November')}}",
                        "{{__('December')}}"
                    ],
                }
            }, cb);

            cb(start, end);
        });
        $(document).on('change', "select[name=project_id]", function () {
            $.get('@auth('web'){{route('home')}}@elseauth{{route('client.home')}}@endauth' + '/userProjectJson/' + $(this).val(), function (data) {
                $('select[name=assign_to]').html('');
                data = JSON.parse(data);
                $(data).each(function (i, d) {
                    $('select[name=assign_to]').append('<option value="' + d.id + '">' + d.name + ' - ' + d.email + '</option>');
                });
            });
            $.get('@auth('web'){{route('home')}}@elseauth{{route('client.home')}}@endauth' + '/projectMilestoneJson/' + $(this).val(), function (data) {
                $('select[name=milestone_id]').html('<option value="">{{__('Select Milestone')}}</option>');
                data = JSON.parse(data);
                $(data).each(function (i, d) {
                    $('select[name=milestone_id]').append('<option value="' + d.id + '">' + d.title + '</option>');
                });
            })
        })
    </script>

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


<script>
    $(function(){
        $("#submit").click(function() {
            var user =  $("#assign_to option:selected").length;
            if(user == 0){
            $('#user_validation').removeClass('d-none')
                return false;
            }else{
            $('#user_validation').addClass('d-none')
            }
        });
    });
</script>
