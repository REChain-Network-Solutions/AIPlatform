@php
    $is_active = false;

    foreach ($item['children'] as $child) {
        if (!Route::has($child['route'])) {
            continue;
        }

        $child_href = $child['route_slug'] ? route($child['route'], $child['route_slug']) : route($child['route']);

        if ($child_href === url()->current()) {
            $is_active = true;
            break;
        }
    }

    $storage_key = 'lqd-menu-collapse-' . data_get($item, 'key', '');
@endphp

<li
    class="lqd-navbar-item lqd-navbar-label-collapsible group/nav-item relative"
    x-data="{
        isOpen: {{ $is_active ? 'true' : "localStorage.getItem('" . $storage_key . "') !== '0'" }},
        toggle() {
            this.isOpen = !this.isOpen;
            localStorage.setItem('{{ $storage_key }}', this.isOpen ? '1' : '0');
        }
    }"
>
    <span
        class="lqd-navbar-label-wrap flex w-full min-w-0 cursor-pointer items-center gap-2 pb-navbar-link-pb pe-navbar-link-pe ps-navbar-link-ps pt-navbar-link-pt lg:group-[&.navbar-shrinked]/body:hidden"
        @click="toggle()"
    >
        <span
            class="lqd-navbar-label inline-block min-w-0 max-w-full flex-1 overflow-hidden text-ellipsis text-4xs uppercase tracking-widest lg:group-[&.navbar-shrinked]/body:w-full lg:group-[&.navbar-shrinked]/body:px-2 lg:group-[&.navbar-shrinked]/body:text-center"
        >
            {{ __(data_get($item, 'label')) }}
        </span>

        @if (!empty(data_get($item, 'badge')))
            <x-badge
                class="shrink-0 rounded-md text-[0.5625rem] group-[&.navbar-shrinked]/body:hidden"
                variant="secondary"
            >
                {{ mb_strtoupper(data_get($item, 'badge')) }}
            </x-badge>
        @endif

        <span
            class="lqd-nav-label-chevron shrink-0 transition-transform group-[&.navbar-shrinked]/body:hidden"
            :class="{ 'rotate-180': isOpen }"
        >
            <x-tabler-chevron-down
                class="w-3"
                stroke-width="2.5"
            />
        </span>
    </span>

    <ul
        class="lqd-navbar-label-children lg:group-[&.navbar-shrinked]/body:!block lg:group-[&.navbar-shrinked]/body:!h-auto lg:group-[&.navbar-shrinked]/body:!overflow-visible"
        x-show="isOpen"
        x-collapse
    >
        @foreach ($item['children'] as $child)
            @php
                $childKey = data_get($child, 'key');
            @endphp

            @if (\App\Helpers\Classes\PlanHelper::planMenuCheck($userPlan, $childKey))
                @if (data_get($child, 'show_condition', true) && data_get($item, 'is_active'))
                    @php
                        $child_href =
                            $child['route_slug'] && \App\Helpers\Classes\Helper::hasRoute($child['route'])
                                ? route($child['route'], $child['route_slug'])
                                : route(\App\Helpers\Classes\Helper::hasRoute($child['route']) ? $child['route'] : 'default');
                        $child_is_active = $child_href === url()->current();
                    @endphp

                    <x-navbar.item id="{{ data_get($item, 'key') }}-{{ data_get($child, 'key') }}">
                        <x-navbar.link
                            class="{{ data_get($child, 'class') }}"
                            class:letter-icon="{{ $child['letter_icon_bg'] ?? '' }}"
                            letter-icon-styles="{{ $child['letter_icon_bg'] ?? '' }}"
                            label="{!! __($child['label']) !!}"
                            href="{{ $child['route'] }}"
                            slug="{{ $child['route_slug'] }}"
                            icon="{{ $child['icon'] }}"
                            active-condition="{{ $child_is_active }}"
                            letter-icon="{{ (int) ($child['letter_icon'] ?? 0) }}"
                            onclick="{{ data_get($child, 'onclick') ?? '' }}"
                            badge="{{ data_get($child, 'badge') ?? '' }}"
                        />
                    </x-navbar.item>
                @endif
            @endif
        @endforeach
    </ul>
</li>
