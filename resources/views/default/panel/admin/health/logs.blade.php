@extends('panel.layout.settings')
@section('title', __('Site Logs'))
@section('titlebar_actions', '')

@section('settings')
    @if (Auth::user()?->isAdmin())
        @if ($app_is_not_demo)
            <div class="">
                <label
                    class="!mb-5 block"
                    for="log"
                >{{ __('You can copy the below info as simple text with Ctrl+C / Ctrl+V:') }}</label>
                <textarea
                    class="form-control w-full"
                    id="log"
                    name="log"
                    cols="30"
                    rows="20"
                >{{ $logContent ?? __('No logged any data.') }}</textarea>

                <div class="!mt-5 space-x-2">
                    <x-button
                        id="copyButton"
                        variant="outline"
                    >
                        <svg
                            class="me-2"
                            xmlns="http://www.w3.org/2000/svg"
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            fill="none"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        >
                            <path
                                stroke="none"
                                d="M0 0h24v24H0z"
                                fill="none"
                            ></path>
                            <path d="M8 8m0 2a2 2 0 0 1 2 -2h8a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-8a2 2 0 0 1 -2 -2z"></path>
                            <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
                        </svg>
                        {{ __('Copy') }}
                    </x-button>
                </div>
            </div>
        @endif

    @endif
@endsection

@push('script')
    <script>
        var logTextarea = document.getElementById('log');

        if (logTextarea) {
            logTextarea.addEventListener('click', function() {
                this.select();
            });
        }

        var copyButton = document.getElementById('copyButton');

        if (copyButton && logTextarea) {
            copyButton.addEventListener('click', function() {
                var done = function() {
                    toastr.success(@json(__('Copied')));
                };

                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(logTextarea.value).then(done).catch(function() {
                        logTextarea.select();
                        document.execCommand('copy');
                        done();
                    });
                } else {
                    logTextarea.select();
                    document.execCommand('copy');
                    done();
                }
            });
        }
    </script>
@endpush
