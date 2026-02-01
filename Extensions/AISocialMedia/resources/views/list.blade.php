@extends('panel.layout.app')
@section('title', 'Manage Scheduled Posts')
@section('titlebar_actions', '')

@push('css')
    <link
        href="{{ custom_theme_url('/assets/libs/datepicker/air-datepicker.css') }}"
        rel="stylesheet"
    />
    <script src="{{ custom_theme_url('/assets/libs/datepicker/air-datepicker.js') }}"></script>
    <script src="{{ custom_theme_url('/assets/libs/datepicker/locale/en.js') }}"></script>

    <link
        href="{{ custom_theme_url('/assets/libs/picker/picker.css') }}"
        rel="stylesheet"
    />
    <script src="{{ custom_theme_url('/assets/libs/picker/picker.js') }}"></script>

    <style>
        .air-datepicker {
            width: 100% !important;
        }

        .air-datepicker-cell.-selected- {
            background: #330582 !important;
        }

        .air-datepicker-cell {
            border-radius: 50% !important;
            width: 32px !important;
            justify-self: center !important;
        }

        .air-datepicker.-inline- {
            border-color: lavenderblush !important;
        }
    </style>
@endpush

@section('content')
    <div class="py-10">
        <p>
            {{ __('Current System Time:') }} <strong>{{ now()->format('H:i:s') }}</strong>
        </p>
        <x-table>
            <x-slot:head>
                <tr>
                    <th>{{ __('Platform') }}</th>
                    <th>{{ __('Products/Services') }}</th>
                    <th>{{ __('Campagin') }}</th>
                    <th>{{ __('Schedule Time') }}</th>
                    <th>{{ __('Period') }}</th>
                    <th>{{ __('Action') }}</th>
                </tr>
            </x-slot:head>

            <x-slot:body>
                @forelse (auth()->user()->scheduledPosts ?? [] as $entry)
                    @php
                        $services = [];
                        $services = explode(',', str_replace(['(', ')'], '', $entry?->products));
                    @endphp

                    <tr>
                        <td> {{ $entry?->platform == 1 ? 'Twitter/X' : ($entry->platform == 2 ? 'LinkedIn' : 'Instagram') }} </td>

                        <td>
                            @foreach ($services ?? [] as $service)
                                @php
                                    [$key, $value] = explode(':', $service);
                                @endphp
                                <p class="m-0">{{ $key }}: {{ $value }}</p>
                            @endforeach
                        </td>

                        <td>{{ $entry?->campaign_name }}</td>

                        <td>{{ $entry?->repeat_time }}</td>

                        <td>
                            @switch($entry?->repeat_period)
                                @case('day')
                                    Daily
                                @break

                                @case('month')
                                    Monthly
                                @break

                                @case('week')
                                    Weekly
                                @break

                                @default
                                    unknow
                            @endswitch

                        </td>

                        <td class="whitespace-nowrap">
                            @if (env('APP_STATUS') == 'Demo')
                                <x-button
                                    class="size-9"
                                    variant="ghost-shadow"
                                    size="none"
                                    onclick="return toastr.info('This feature is disabled in Demo version.')"
                                    title="{{ __('Edit') }}"
                                >
                                    <x-tabler-pencil class="size-4" />
                                </x-button>
                                <x-button
                                    class="size-9"
                                    variant="ghost-shadow"
                                    hover-variant="danger"
                                    size="none"
                                    onclick="return toastr.info('This feature is disabled in Demo version.')"
                                    title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="size-4" />
                                </x-button>
                            @else
                                <x-button
                                    class="size-9 edit-button"
                                    data-id="{{ $entry->id }}"
                                    data-platform="{{ $entry->platform }}"
                                    data-period="{{ $entry->repeat_period }}"
                                    data-time="{{ $entry->repeat_time }}"
                                    data-date="{{ $entry->repeat_start_date }}"
                                    data-repeated="{{ $entry->is_repeated }}"
                                    data-bs-toggle="modal"
                                    data-bs-target="#editModal"
                                    variant="ghost-shadow"
                                    size="none"
                                    title="{{ __('Edit') }}"
                                >
                                    <x-tabler-pencil class="size-4" />
                                </x-button>
                                <x-button
                                    class="size-9 delete-button"
                                    variant="ghost-shadow"
                                    hover-variant="danger"
                                    size="none"
                                    href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.automation.delete', $entry->id)) }}"
                                    onclick="return  confirm('{{ __('Are you sure? This is permanent.') }}')"
                                    title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="size-4" />
                                </x-button>
                            @endif
                        </td>
                    </tr>
                    @empty
                        <tr>
                            <td
                                class="text-center"
                                colspan="8"
                            >
                                {{ __('There is no scheduled posts yet') }}
                            </td>
                        </tr>
                    @endforelse
                </x-slot:body>
            </x-table>
        </div>

        <!-- Editing post Modal -->
        <div
            class="modal fade"
            id="editModal"
            tabindex="-1"
            aria-labelledby="editModalLabel"
            aria-hidden="true"
        >
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5
                            class="modal-title"
                            id="editModalLabel"
                        >{{ __('Edit Scheduled Post') }}</h5>
                    </div>
                    <form
                        id="editForm"
                        method="post"
                    >
                        <div class="modal-body">
                            @csrf

                            <div class="row mb-[15px]">
                                <div class="mb-3 flex flex-wrap content-between">
                                    <div class="flex flex-1 items-start justify-start gap-2 px-4 pb-0 pt-3">
                                        <label class="form-check form-switch">
                                            <input
                                                class="form-check-input"
                                                id="mrepeated"
                                                name="repeat"
                                                type="checkbox"
                                            >
                                        </label>
                                        <div class="font-['Inter V'] m-0 self-center text-sm font-medium leading-none text-neutral-900">{{ __('Repeat?') }}
                                            <x-info-tooltip text="{{ __('') }}" />
                                        </div>
                                    </div>

                                    <select
                                        class="form-select align-self-end mx-4 w-auto flex-1 content-end justify-self-end pe-4"
                                        id="mperiod"
                                        style="height: fit-content;"
                                        name="repeat_period"
                                        required
                                    >
                                        <option value="day">{{ __('Every Day') }}</option>
                                        <option value="week">{{ __('Every Week') }}</option>
                                        <option value="month">{{ __('Every Month') }}</option>
                                    </select>
                                </div>
                            </div>

                            <div class="row mb-[15px]">
                                <div
                                    class="w-100 mb-3 inline-flex flex-col items-center justify-start gap-3 px-4"
                                    id="el"
                                >
                                    <input
                                        id="hiddenDateInput"
                                        type="hidden"
                                        name="date"
                                    >
                                </div>

                            </div>

                            <div class="row mb-[15px]">

                                <div class="w-100 mb-3 inline-flex flex-col items-start justify-start gap-3 px-4">
                                    <div class="inline-flex items-center justify-start gap-2">
                                        <div class="flex items-center justify-center gap-2 rounded">
                                            <div class="font-['Inter V'] text-base font-medium leading-none text-neutral-900">{{ __('Time') }} ({{ config('app.timezone') }})

                                            </div>
                                        </div>
                                    </div>
                                    <input
                                        class="form-control js-time-picker"
                                        id="mtime"
                                        type="text"
                                        name="time"
                                        value="02:56"
                                    >
                                </div>
                            </div>

                        </div>
                        <div class="modal-footer">
                            <button
                                class="btn btn-secondary"
                                data-bs-dismiss="modal"
                                type="button"
                            >{{ __('Close') }}</button>
                            <button
                                class="btn btn-primary"
                                type="submit"
                            >{{ __('Save changes') }}</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    @endsection

    @push('script')
        <script src="{{ custom_theme_url('/assets/libs/datepicker/air-datepicker.js') }}"></script>
        <script src="{{ custom_theme_url('/assets/libs/datepicker/locale/en.js') }}"></script>
        <script src="{{ custom_theme_url('/assets/libs/picker/picker.js') }}"></script>
        <script>
            $(document).ready(function() {
                "use strict";

                var airDatepicker = new AirDatepicker('#el', {
                    locale: defaultLocale,
                    minDate: new Date(),
                    onSelect({
                        date,
                        formattedDate,
                        datepicker
                    }) {
                        const hiddenInput = document.getElementById('hiddenDateInput');
                        hiddenInput.value = formattedDate;
                    }
                });


                new Picker(document.querySelector('.js-time-picker'), {
                    format: 'HH:mm',
                    headers: true,
                    text: {
                        title: '{{ __('Pick a time') }}',
                    },
                });


                $('.edit-button').on('click', function() {
                    var rowId = $(this).data('id');
                    var period = $(this).data('period');

                    var time = $(this).data('time');
                    console.log(time);
                    var timeWithoutSeconds = time.split(':').slice(0, 2).join(':');

                    var date = $(this).data('date');
                    var repeated = $(this).data('repeated');

                    $('#mperiod').val(period);
                    $('#mtime').val(timeWithoutSeconds);
                    airDatepicker.selectDate(new Date(date));
                    repeated == 1 ? $('#mrepeated').prop('checked', true) : $('#mrepeated').prop('checked', false);

                    var editForm = $('#editForm');
                    var actionUrl = "{{ route('dashboard.user.automation.edit', ':id') }}";
                    actionUrl = actionUrl.replace(':id', rowId);
                    editForm.attr('action', actionUrl);
                });

            });
        </script>
    @endpush
