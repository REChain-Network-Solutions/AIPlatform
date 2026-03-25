@extends('layouts.app')
@push('styles')
    <link rel="stylesheet" href="{{ asset('vendor/full-calendar/main.min.css') }}">
    <link rel="stylesheet" href="{{ asset('vendor/css/bootstrap-colorpicker.css') }}" />
@endpush

@section('filter-section')
@endsection

@section('content')
    <div class="content-wrapper">
        <x-cards.data>
            <div id="calendar"></div>
        </x-cards.data>
    </div>
@endsection

@push('scripts')
    <script src="{{ asset('vendor/full-calendar/main.min.js') }}"></script>
    <script src="{{ asset('vendor/full-calendar/locales-all.min.js') }}"></script>

    <script>
        var initialLocaleCode = '{{ user()->locale }}';
        var calendarEl = document.getElementById('calendar');

        var calendar = new FullCalendar.Calendar(calendarEl, {
            locale: initialLocaleCode,
            timeZone: '{{ company()->timezone }}',
            headerToolbar: {
                left: 'prev,next today',
                center: 'title',
                right: 'dayGridMonth,timeGridWeek,timeGridDay,listWeek'
            },
            navLinks: true, // can click day/week names to navigate views
            selectable: false,
            selectMirror: true,
            select: function(arg) {
                addEventModal(arg.start, arg.end, arg.allDay);
                calendar.unselect()
            },
            eventClick: function(arg) {
                getEventDetail(arg.event.id);
            },
            editable: false,
            dayMaxEvents: true, // allow "more" link when too many events
            events: {!! json_encode($eventData) !!},
            eventDidMount: function(info) {
                // $(info.el).css('background-color', info.event.extendedProps.bg_color);
                // $(info.el).css('color', info.event.extendedProps.color);
            },
            eventTimeFormat: {
                hour: company.time_format == 'H:i' ? '2-digit' : 'numeric',
                minute: '2-digit',
                hour12: company.time_format == 'H:i' ? false : true,
                meridiem: company.time_format == 'H:i' ? false : true
            }
        });

        calendar.render();

        function loadData() {
            calendar.refetchEvents();
            calendar.destroy();
            calendar.render();
        }

        // show event detail in sidebar
        var getEventDetail = function(id) {
            var url = "{{ route('work.show', ':id') }}";
            url = url.replace(':id', id);

            window.location.href = url;
        }
    </script>
@endpush
