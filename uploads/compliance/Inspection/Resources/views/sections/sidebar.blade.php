@if (in_array('inspections', user_modules()))
    <x-menu-item icon="camera-video" :text="__('inspection::sidebar.inspection')" :addon="App::environment('schedule')">
        <x-slot name="iconPath">
            <path fill-rule="evenodd"
                  d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2V5zm11.5 5.175 3.5 1.556V4.269l-3.5 1.556v4.35zM2 4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h7.5a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1H2z"/>
        </x-slot>
        <div class="accordionItemContent pb-2">
            @php
                // Guard against RouteNotFoundException during partial installs/upgrades.
                $rRecurring = Route::has('recurring-inspection_schedules.index')
                    ? route('recurring-inspection_schedules.index')
                    : (Route::has('recurring-schedules.index') ? route('recurring-schedules.index') : null);

                $rSchedules = Route::has('inspection_schedules.index')
                    ? route('inspection_schedules.index')
                    : (Route::has('schedules.index') ? route('schedules.index') : null);

                $rInspections = Route::has('schedule-inspection.index')
                    ? route('schedule-inspection.index')
                    : null;
            @endphp

            @if($rRecurring)
                <x-sub-menu-item :link="$rRecurring" :text="__('inspection::sidebar.recurring')"/>
            @endif

            @if($rSchedules)
                <x-sub-menu-item :link="$rSchedules" :text="__('inspection::sidebar.schedules')"/>
            @endif

            @if($rInspections)
                <x-sub-menu-item :link="$rInspections" :text="__('inspection::sidebar.inspections')"/>
            @endif
        </div>
    </x-menu-item>
@endif
