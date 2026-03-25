@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Pipelines — CRM'))
@section('titlebar_title', __('Pipelines'))
@section('titlebar_subtitle', __('Manage your deal pipelines and stages.'))

@section('titlebar_actions')
    <x-button href="{{ route('dashboard.crm.pipelines.create') }}" size="sm" variant="primary">
        <x-tabler-plus class="w-4" />
        {{ __('New Pipeline') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        @if (session('success'))
            <div class="mb-6 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
                {{ session('success') }}
            </div>
        @endif

        @if ($pipelines->isEmpty())
            <div class="py-20 text-center text-muted-foreground">
                <x-tabler-git-branch class="mx-auto mb-4 w-12 opacity-30" />
                <p class="mb-3">{{ __('No pipelines yet.') }}</p>
                <x-button href="{{ route('dashboard.crm.pipelines.create') }}" size="sm" variant="outline">
                    {{ __('Create your first pipeline') }}
                </x-button>
            </div>
        @else
            <div class="grid grid-cols-1 gap-5 lg:grid-cols-2">
                @foreach ($pipelines as $pipeline)
                    <div class="rounded-2xl border border-border bg-card shadow-xs">
                        <div class="flex items-center justify-between border-b border-border px-5 py-4">
                            <div class="flex items-center gap-3">
                                @if ($pipeline->is_default)
                                    <span class="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">{{ __('Default') }}</span>
                                @endif
                                <h3 class="font-semibold text-heading-foreground">{{ $pipeline->name }}</h3>
                                <span class="text-xs text-muted-foreground">{{ $pipeline->currency }}</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-xs text-muted-foreground">{{ $pipeline->deals_count }} {{ __('deals') }}</span>
                                <a href="{{ route('dashboard.crm.pipelines.edit', $pipeline) }}" class="text-muted-foreground hover:text-primary" title="{{ __('Edit') }}">
                                    <x-tabler-pencil class="w-4" />
                                </a>
                                @if ($pipeline->deals_count === 0)
                                    <form method="POST" action="{{ route('dashboard.crm.pipelines.destroy', $pipeline) }}" onsubmit="return confirm('{{ __('Delete this pipeline?') }}')">
                                        @csrf @method('DELETE')
                                        <button type="submit" class="text-muted-foreground hover:text-red-500" title="{{ __('Delete') }}">
                                            <x-tabler-trash class="w-4" />
                                        </button>
                                    </form>
                                @endif
                            </div>
                        </div>

                        {{-- Stages --}}
                        <div class="p-4">
                            @if ($pipeline->stages->isEmpty())
                                <p class="text-center text-xs text-muted-foreground py-4">{{ __('No stages — edit to add some.') }}</p>
                            @else
                                <div class="flex items-center gap-0">
                                    @foreach ($pipeline->stages as $i => $stage)
                                        <div class="group relative flex-1">
                                            <div
                                                class="flex flex-col items-center gap-1 px-1 py-2 text-center"
                                                title="{{ $stage->probability }}% probability"
                                            >
                                                <div class="h-2 w-full rounded-sm" style="background-color: {{ $stage->color ?? '#6366f1' }}"></div>
                                                <span class="text-xs text-heading-foreground leading-tight mt-1 truncate w-full text-center">{{ $stage->name }}</span>
                                                @if ($stage->is_won)
                                                    <span class="text-xs text-green-600">{{ __('Won') }}</span>
                                                @elseif ($stage->is_lost)
                                                    <span class="text-xs text-red-500">{{ __('Lost') }}</span>
                                                @else
                                                    <span class="text-xs text-muted-foreground">{{ $stage->probability }}%</span>
                                                @endif
                                            </div>
                                        </div>
                                        @if (!$loop->last)
                                            <x-tabler-chevron-right class="w-3 shrink-0 text-muted-foreground/40" />
                                        @endif
                                    @endforeach
                                </div>
                            @endif
                        </div>
                    </div>
                @endforeach
            </div>
        @endif

    </div>
@endsection
