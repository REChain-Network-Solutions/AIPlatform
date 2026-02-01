@extends('panel.layout.app')
@section('title', __('Chat Templates'))

@section('titlebar_actions')
    <x-button
            href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.chat-setting.chat-template.create')) }}"
    >
        {{ __('Add Template') }}
    </x-button>
@endsection

@section('content')
    <!-- Page body -->
    <div class="py-6">
        <div class="container-xl">
            <div class="card">
                <div id="table-default" class="card-table table-responsive">
                    <table class="table">
                        <thead>
                        <tr>
                            <th>{{ __('Template Name') }}</th>
                            <th>{{ __('Template Description') }}</th>
                            <th>{{ __('Role') }}</th>
                            <th>{{ __('Plan') }}</th>
                            <th>{{ __('Updated At') }}</th>
                        </tr>
                        </thead>
                        <tbody class="align-middle table-tbody text-heading">
                        @foreach ($list as $entry)
                            <tr>
                                <td class="sort-name">{{ $entry->name }}</td>
                                <td class="sort-description">{{ $entry->description }}</td>
                                <td class="sort-description">{{ $entry->role }}</td>
                                <td class="sort-package">
                                    @if ($app_is_demo)
                                        <select class="form-control min-w-[90px]"
                                                onchange="return toastr.info('This feature is disabled in Demo version.');"
                                                name="premium" id="premium">
                                            @else
                                                <select class="form-control min-w-[90px]"
                                                        onchange="return updateChatStatus(this.value, {{ $entry->id }});"
                                                        name="premium" id="premium">
                                                    @endif
                                                    <option value="premium" {{ $entry->plan == 'premium' ? 'selected' : '' }}>
                                                        {{ __('Premium') }} </option>
                                                    <option value="regular" {{ $entry->plan == 'premium' ? '' : 'selected' }}>
                                                        {{ __('Regular') }}</option>
                                                </select>

                                </td>

                                <td class="sort-date" data-date="{{ strtotime($entry->updated_at) }}">
                                    <p class="m-0">{{ date('j.n.Y', strtotime($entry->updated_at)) }}</p>
                                    <p class="m-0 text-muted">{{ date('H:i:s', strtotime($entry->updated_at)) }}</p>
                                </td>
                                <td class="whitespace-nowrap">
									<a href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.chat-setting.chat-template.edit', $entry->id)) }}"
									   class="btn w-[36px] h-[36px] p-0 border hover:bg-[var(--tblr-primary)] hover:text-white"
									   title="{{ __('Edit') }}">
										<svg width="13" height="12" viewBox="0 0 16 15" fill="none"
											 stroke="currentColor" xmlns="http://www.w3.org/2000/svg">
											<path
												d="M9.3125 2.55064L12.8125 5.94302M11.5 12.3038H15M4.5 14L13.6875 5.09498C13.9173 4.87223 14.0996 4.60779 14.224 4.31676C14.3484 4.02572 14.4124 3.71379 14.4124 3.39878C14.4124 3.08377 14.3484 2.77184 14.224 2.48081C14.0996 2.18977 13.9173 1.92533 13.6875 1.70259C13.4577 1.47984 13.1849 1.30315 12.8846 1.1826C12.5843 1.06205 12.2625 1 11.9375 1C11.6125 1 11.2907 1.06205 10.9904 1.1826C10.6901 1.30315 10.4173 1.47984 10.1875 1.70259L1 10.6076V14H4.5Z"
												stroke-width="1.25" stroke-linecap="round"
												stroke-linejoin="round" />
										</svg>
									</a>
                                    @if ($app_is_demo)
                                        <a onclick="return toastr.info('This feature is disabled in Demo version.')"
                                           class="btn p-0 border w-[36px] h-[36px] hover:bg-red-600 hover:text-white"
                                           title="{{ __('Delete') }}">
                                            <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"
                                                 xmlns="http://www.w3.org/2000/svg">
                                                <path
                                                        d="M9.08789 1.74609L5.80664 5L9.08789 8.25391L8.26758 9.07422L4.98633 5.82031L1.73242 9.07422L0.912109 8.25391L4.16602 5L0.912109 1.74609L1.73242 0.925781L4.98633 4.17969L8.26758 0.925781L9.08789 1.74609Z" />
                                            </svg>
                                        </a>
                                    @else
                                        <a href="#"
                                           onclick="deleteChatbot('{{ route('dashboard.user.chat-setting.chat-template.destroy', $entry->id) }}')"
                                           class="btn w-[36px] h-[36px] p-0 border hover:bg-red-500 hover:text-white"
                                           title="{{ __('Delete') }}">
                                            <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"
                                                 xmlns="http://www.w3.org/2000/svg">
                                                <path
                                                        d="M9.08789 1.74609L5.80664 5L9.08789 8.25391L8.26758 9.07422L4.98633 5.82031L1.73242 9.07422L0.912109 8.25391L4.16602 5L0.912109 1.74609L1.73242 0.925781L4.98633 4.17969L8.26758 0.925781L9.08789 1.74609Z" />
                                            </svg>
                                        </a>
                                    @endif
                                </td>
                            </tr>
                        @endforeach

                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('script')
    <script src="/assets/js/panel/openai.js"></script>
    <script>
        function deleteChatbot(url) {
            if (confirm('{{ __('Are you sure? This is chatbot and will delete.') }}')) {
                $.ajax({
                    type: "DELETE",
                    url: url,
                    success: function(result) {
                        if (result.status == 'error') {
                            toastr.error(result.message);
                        } else {
                            toastr.success(result.message);
                            if (result.reload) {
                                setTimeout(function() {
                                    location.reload();
                                }, result.setTimeOut);
                            }
                        }
                    }
                });
            }
        }
    </script>
@endpush
