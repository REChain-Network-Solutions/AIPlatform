<x-card
    class="p-6"
    class:body="flex flex-col gap-6 p-0"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'what-is-new') }}"
    size="md"
>
    <x-slot:head
        class="border-b-0 p-0 pb-4"
    >
        <h2 class="font-bold">@lang("What's New")</h2>
    </x-slot:head>
    <div class="grid grid-cols-2 gap-6 sm:gap-4 xl:grid-cols-3 2xl:gap-8">
        <x-card
            class="relative grow cursor-pointer"
            class:body="flex flex-col gap-6"
            size="sm"
        >
            <h2 class="sm:text-[30px]">{{ cache()->get('tickets') ?? 0 }}</h2>
            <div class="flex flex-col gap-3">
                <x-tabler-lifebuoy class="size-6"></x-tabler-lifebuoy>
                <p class="text-lg font-medium">@lang('Tickets')</p>
            </div>
            <a
                class="absolute inset-0"
                href="{{ route('dashboard.support.list') }}"
            ></a>
        </x-card>
        <x-card
            class="relative grow cursor-pointer"
            class:body="flex flex-col gap-6"
            size="sm"
        >
            <h2 class="sm:text-[30px]">{{ cache()->get('transactions') ?? 0 }}</h2>
            <div class="flex flex-col gap-3">
                <x-tabler-coins class="size-6"></x-tabler-coins>
                <p class="text-lg font-medium">@lang('Transactions')</p>
            </div>
            <a
                class="absolute inset-0"
                href="{{ route('dashboard.admin.users.index') }}"
            ></a>
        </x-card>
        <x-card
            class="relative grow cursor-pointer"
            class:body="flex flex-col gap-6"
            size="sm"
        >
            <h2 class="sm:text-[30px]">{{ cache()->get('documents') ?? 0 }}</h2>
            <div class="flex flex-col gap-3">
                <x-tabler-clipboard-text class="size-6"></x-tabler-clipboard-text>
                <p class="text-lg font-medium">@lang('Documents')</p>
            </div>
            <a
                class="absolute inset-0"
                href="{{ route('dashboard.user.openai.documents.all') }}"
            ></a>
        </x-card>
    </div>
    <x-card
        class="w-full"
        class:body="flex items-center justify-between py-10 px-6"
        size="lg"
    >
        <div class="flex flex-col gap-0.5">
            <h3 class="font-semibold">{{ __('Magic AI is Up-to-date.') }}</h3>
            <p class="mb-0">{{ __('Version') }} {{ format_double($setting->script_version) }}</p>
        </div>
        <x-tabler-checks
            class="size-12"
            stroke="url(#paint0_linear_9208_560_0)"
            stroke-width="1"
        ></x-tabler-checks>
    </x-card>
</x-card>
