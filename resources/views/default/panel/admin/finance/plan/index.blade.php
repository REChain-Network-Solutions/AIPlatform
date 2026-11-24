@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Subscriptions and Packs'))
@section('titlebar_actions')
    <div class="flex flex-wrap gap-3">
        <x-dropdown.dropdown
            anchor="end"
            offsetY="10px"
        >
            <x-slot:trigger variant="primary">
                <x-tabler-plus class="size-4" />
                {{ __('Create Plan') }}
                <x-tabler-chevron-down class="size-4" />
            </x-slot:trigger>

            <x-slot:dropdown class="min-w-fit overflow-hidden whitespace-nowrap py-1 text-2xs font-medium">
                <x-button
                    class="flex w-full justify-start rounded-none px-3 py-2 text-2xs shadow-none hover:translate-y-0 hover:bg-foreground/5 hover:text-inherit hover:shadow-none focus-visible:bg-foreground/5 focus-visible:text-inherit"
                    size="none"
                    variant="ghost-shadow"
                    @click.prevent="Alpine.$data(document.querySelector('#ai-plan-wizard-root'))?.openWizard()"
                >
                    <x-tabler-sparkles class="size-4" />
                    {{ __('Create with AI') }}
                </x-button>
                <x-button
                    class="flex w-full justify-start rounded-none px-3 py-2 text-2xs shadow-none hover:translate-y-0 hover:bg-foreground/5 hover:text-inherit hover:shadow-none focus-visible:bg-foreground/5 focus-visible:text-inherit"
                    size="none"
                    variant="ghost-shadow"
                    href="{{ $app_is_demo ? '#' : route('dashboard.admin.finance.plan.create') }}"
                    onclick="{{ $app_is_demo ? 'return toastr.info(\'This feature is disabled in Demo version.\')' : '' }}"
                >
                    <x-tabler-repeat class="size-4" />
                    {{ __('Create New Subscription') }}
                </x-button>
                <x-button
                    class="flex w-full justify-start rounded-none px-3 py-2 text-2xs shadow-none hover:translate-y-0 hover:bg-foreground/5 hover:text-inherit hover:shadow-none focus-visible:bg-foreground/5 focus-visible:text-inherit"
                    size="none"
                    variant="ghost-shadow"
                    href="{{ $app_is_demo ? '#' : route('dashboard.admin.finance.token-pack-plan.create', ['type' => 'token-pack']) }}"
                    onclick="{{ $app_is_demo ? 'return toastr.info(\'This feature is disabled in Demo version.\')' : '' }}"
                >
                    <x-tabler-coins class="size-4" />
                    {{ __('Create New Token Pack') }}
                </x-button>
            </x-slot:dropdown>
        </x-dropdown.dropdown>
    </div>
@endsection
@section('content')
    <div class="py-10">
        @if ($settings_two->liquid_license_type !== 'Extended License')
            <div class="rounded-xl bg-red-500/20 p-3 text-center text-red-600 dark:text-red-300">
                {{ __('To access this page, you should upgrade to Extended License.') }}
                <a href="{{ route('LaravelInstaller::license.upgrade') }}">
                    <u>
                        {{ __('Upgrade License') }}
                    </u>
                </a>
            </div>
        @else
            @if ($gatewayError == true)
                <div class="mt-2 rounded-xl bg-amber-100 p-4 text-amber-600 dark:bg-amber-600/20 dark:text-amber-200">
                    <div class="mb-5 flex flex-wrap items-center gap-2">
                        <x-tabler-info-circle class="size-5" />
                        {{ __('Gateway is set to use sandbox. Please set mode to development!') }}
                    </div>
                    <ul class="list-inside list-disc">
                        <li>
                            {{ __('To use live settings:') }}
                            <ul class="list-inside list-disc ps-4">
                                <li>{{ __('Set mode to Production') }}</li>
                                <li>{{ __('Save gateway settings') }}</li>
                                <li>{{ __('Know that all defined products and prices will reset.') }}</li>
                            </ul>
                        </li>
                        <li>
                            {{ __('To use sandbox settings:') }}
                            <ul class="list-inside list-disc ps-4">
                                <li>{{ __('Set mode to Development') }}</li>
                                <li>{{ __('Save gateway settings') }}</li>
                                <li>{{ __('Know that all defined products and prices will reset.') }}</li>
                            </ul>
                        </li>
                        <li>{{ __('Beware of that order is important. First set mode then save gateway settings.') }}</li>
                    </ul>
                </div>
            @endif

            <x-table>
                <x-slot:head>
                    <tr>
                        <th>
                            {{ __('Status') }}
                        </th>
                        <th>
                            {{ __('Type') }}
                        </th>
                        <th>
                            {{ __('Name') }}
                        </th>
                        <th>
                            {{ __('Price') }}
                        </th>
                        <th>
                            {{ __('Frequency') }}
                        </th>
                        <th>
                            {{ __('Updated At') }}
                        </th>
                        <th class="text-end">
                            {{ __('Actions') }}
                        </th>
                    </tr>
                </x-slot:head>

                <x-slot:body>
                    @foreach ($plans as $entry)
                        <tr>
                            <td>
                                <x-badge
                                    class="text-2xs"
                                    variant="{{ $entry->active == 1 ? 'success' : 'danger' }}"
                                >
                                    {{ $entry->active == 1 ? __('Active') : __('Passive') }}
                                </x-badge>
                            </td>
                            <td>
                                {{ $entry->type === 'prepaid' ? __('Token Pack') : __('Subscription') }}
                            </td>
                            <td>
                                {{ $entry->name }}
                                @if($entry->isSharedCreditPlan())
                                    <x-badge
                                        class="text-2xs ms-1"
                                        variant="info"
                                    >
                                        {{ __('Shared') }}
                                    </x-badge>
                                @endif
                            </td>
                            <td>
                                {{ $entry->price }}
                            </td>
                            <td>
                                {{ $entry->type === 'prepaid' ? __('One Time') : __(formatCamelCase($entry->frequency)) }}
                            </td>
                            <td>
                                <p class="m-0">
                                    {{ $entry->updated_at->toDateString() }}
                                    <span class="block opacity-60">
                                        {{ $entry->updated_at->toTimeString() }}
                                    </span>
                                </p>
                            </td>
                            <td class="whitespace-nowrap text-end">
                                <x-button
                                    class="size-9"
                                    variant="ghost-shadow"
                                    size="none"
                                    href="{{ route($entry->type == 'subscription' ? 'dashboard.admin.finance.plan.edit' : 'dashboard.admin.finance.token-pack-plan.edit', $entry->id) }}"
                                    title="{{ __('Edit') }}"
                                >
                                    <x-tabler-pencil class="size-4" />
                                </x-button>
                                @if ($app_is_demo)
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
                                        class="size-9"
                                        variant="ghost-shadow"
                                        hover-variant="danger"
                                        size="none"
                                        onclick="return confirm('Do you want to delete this plan? All subscriptions will be cancelled. This is not reversible.');"
                                        href="{{ route('dashboard.admin.finance.plan.destroy', $entry->id) }}"
                                        title="{{ __('Delete') }}"
                                    >
                                        <x-tabler-x class="size-4" />
                                    </x-button>
                                @endif
                            </td>
                        </tr>
                    @endforeach

                </x-slot:body>
            </x-table>

            @include('panel.admin.finance.plan.ai-wizard.wizard')
        @endif
    </div>
@endsection
