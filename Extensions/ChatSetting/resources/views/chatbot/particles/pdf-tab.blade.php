<div
        class="hidden"
        :class="{ 'hidden': activeTab !== 'pdf' }"
>
    <div
            class="mb-4 flex w-full items-center justify-center"
            id="mainupscale_src"
            ondrop="dropHandler(event, 'upscale_src');"
            ondragover="dragOverHandler(event);"
    >
        <label
                class="min-h-48 flex w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-foreground/10 bg-background text-center text-[12px] transition-colors hover:bg-background/80"
                for="upscale_src"
        >
            <div class="flex flex-col items-center justify-center py-6">
                <x-tabler-circle-plus
                        class="size-9 mb-3.5"
                        stroke-width="1.5"
                />

                <p class="mb-1 font-semibold">
                    {{ __('UPLOAD PDF, XLSX, CSV') }}
                </p>

                <p class="file-name mb-0">
                    {{ __('Upload a File (Max: 25Mb)') }}
                </p>
            </div>

            <input
                    class="hidden"
                    id="upscale_src"
                    data-action="{{ route('dashboard.user.chat-setting.chatbot.upload-pdf', $item->id) }}"
                    name="file"
                    type="file"
{{--                    accept=".pdf"--}}
                    onchange="handleFileSelect('upscale_src')"
            />
        </label>
    </div>

    <form
            class="mt-10 flex flex-col gap-5"
            id="form-train-pdf"
            method="post"
            action="{{ route('dashboard.user.chat-setting.chatbot.training', $item->id) }}"
    >
        <input
                type="hidden"
                name="type"
                value="pdf"
        >

        <div
                class="pdf-list list-common space-y-4"
                id="pdf-list"
        >
            @php
                $pdf = $data->where('type', 'pdf');
            @endphp

            @include('chat-setting::chatbot.particles.pdf.list', ['items' => $pdf])
        </div>

    </form>
</div>
