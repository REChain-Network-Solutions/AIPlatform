@extends('panel.layout.app')
@section('title', __('My Advertis'))

@section('content')
    <div class="py-10">
        <div class="container-xl">
            <h2>{{ __('Advertis Edit') }}</h2>
            <div class="card">
                <div
                    class="card-table table-responsive"
                    id="table-default-2"
                >
                    <form
                        class="m-auto w-1/2 p-4"
                        method="POST"
                        action="{{ route('dashboard.admin.advertis.update', $advertis) }}"
                    >
                        @csrf
                        @method('PUT')
                        <div class="flex flex-col gap-4">
                            <div class="flex w-full flex-col gap-2">
                                <label
                                    class="m-1"
                                    for="name"
                                >Key</label>
                                <input
                                    class="h-10 w-full rounded border-none bg-gray-400 ps-3 focus:border-blue-300"
                                    type="text"
                                    name="key"
                                    value="{{ $advertis->key }}"
                                    disabled
                                >
                            </div>
                            <div class="flex w-full flex-col gap-2">
                                <label
                                    class="m-1"
                                    for="name"
                                >Title</label>
                                <input
                                    class="h-10 w-full rounded border-none bg-gray-400 ps-3 focus:border-blue-300"
                                    type="text"
                                    name="title"
                                    value="{{ old('title', $advertis->title) }}"
                                >
                            </div>
                        </div>
                        <div class="flex flex-col gap-4">
                            <div class="flex w-full flex-col gap-2">
                                <label
                                    class="m-1"
                                    for="name"
                                >Mobile Tracking Code</label>
                                <textarea
                                    class="h-40 w-full rounded border-none bg-gray-400 p-2 focus:border-blue-300"
                                    name="tracking_code[mobile]"
                                >{{ old('tracking_code.mobile', data_get($advertis, 'tracking_code.mobile')) }}</textarea>
                            </div>
                            <div class="flex w-full flex-col gap-2">
                                <label
                                    class="m-1"
                                    for="name"
                                >Tablet Tracking Code</label>
                                <textarea
                                    class="h-40 w-full rounded border-none bg-gray-400 p-2 focus:border-blue-300"
                                    name="tracking_code[tablet]"
                                >{{ old('tracking_code.tablet', data_get($advertis, 'tracking_code.tablet')) }}</textarea>
                            </div>
                            <div class="flex w-full flex-col gap-2">
                                <label
                                    class="m-1"
                                    for="name"
                                >Desktop Tracking Code</label>
                                <textarea
                                    class="h-40 w-full rounded border-none bg-gray-400 p-2 focus:border-blue-300"
                                    name="tracking_code[desktop]"
                                >{{ old('tracking_code.desktop', data_get($advertis, 'tracking_code.desktop')) }}</textarea>
                            </div>
                        </div>

                        <div class="mt-4 flex">
                            <div class="form-check form-switch">
                                <input
                                    class="form-check-input"
                                    id="advertis-status"
                                    type="checkbox"
                                    name="status"
                                    @checked($advertis->status == true)
                                >
                                <label
                                    class="form-check-label"
                                    for="advertis-status"
                                >Advertis Status</label>
                            </div>
                        </div>

                        <div class="flex">
                            <button class="w-full rounded border-none bg-blue-600 p-2 text-center font-semibold text-white">
                                Save
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection
