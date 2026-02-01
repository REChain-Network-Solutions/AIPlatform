@php
    use App\Extensions\Announcement\System\Enum\AnnouncementType;
    $types = AnnouncementType::cases();
@endphp

@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Public Announcements'))

@section('content')
    <div class="py-10">
        <form
            class="mx-auto flex w-full flex-col gap-5 lg:w-5/12"
            id="public_announcement_form"
        >
            @method('post')
            @csrf
            <x-form-step label="{{ __('Add announcement') }}">
                <x-button
                    class="add-more ms-auto inline-flex size-8 items-center justify-center rounded-full bg-background text-foreground transition-all"
                    size="none"
                    type="button"
                    variant="ghost-shadow"
                >
                    <x-tabler-plus class="size-5" />
                </x-button>
            </x-form-step>

            <x-card
                class="repeater-input-group relative"
                class:body="flex flex-col gap-5"
                data-inputs-id="1"
            >
                <x-forms.input
                    class:container="mb-2"
                    class="announce_active"
                    id="active"
                    type="checkbox"
                    label="{{ __('Active') }}"
                    tooltip="{{ __('Enable or disable the announcement.') }}"
                    name="active"
                    switcher
                />

                <x-forms.input
                    class="announce_title"
                    id="title"
                    size="lg"
                    label="{{ __('Title') }}"
                    tooltip="{{ __('The title of the announcement.') }}"
                    placeholder="{{ __('Title') }}"
                    name="title"
                    required
                />

                <x-forms.input
                    class="announce_type"
                    id="type"
                    size="none"
                    name="type"
                    type="select"
                    label="{{ __('Type') }}"
                    tooltip="{{ __('The type of announcement') }}"
                >
                    @foreach ($types as $type)
                        <option value="{{ $type->value }}">
                            {{ $type->label() }}
                        </option>
                    @endforeach
                </x-forms.input>

                <x-button
                    class="remove-inputs-group absolute -end-3 -top-3 size-6"
                    size="none"
                    variant="danger"
                    type="button"
                >
                    <x-tabler-minus class="size-4" />
                </x-button>
            </x-card>

            <div class="add-more-placeholder"></div>

            <x-button
                id="public_announcement_button"
                type="button"
                form="public_announcement_form"
            >
                {{ __('Save') }}
            </x-button>

        </form>

        <template id="repeater-input-announcement">
            <x-card
                class="repeater-input-group relative"
                class:body="flex flex-col gap-5"
                data-inputs-id="1"
            >
                <x-forms.input
                    class:container="mb-2"
                    class="announce_active"
                    id="active"
                    type="checkbox"
                    label="{{ __('Active') }}"
                    tooltip="{{ __('Enable or disable the announcement.') }}"
                    name="active"
                    switcher
                />

                <x-forms.input
                    class="announce_title"
                    id="title"
                    size="lg"
                    label="{{ __('Title') }}"
                    tooltip="{{ __('The title of the announcement.') }}"
                    placeholder="{{ __('Title') }}"
                    name="title"
                    required
                />

                <x-forms.input
                    class="announce_type"
                    id="type"
                    size="none"
                    name="type"
                    type="select"
                    label="{{ __('Type') }}"
                    tooltip="{{ __('The type of announcement') }}"
                >
                    @foreach ($types as $type)
                        <option value="{{ $type->value }}">
                            {{ $type->label() }}
                        </option>
                    @endforeach
                </x-forms.input>

                <x-button
                    class="remove-inputs-group absolute -end-3 -top-3 size-6"
                    size="none"
                    variant="danger"
                    type="button"
                >
                    <x-tabler-minus class="size-4" />
                </x-button>
            </x-card>
        </template>
    </div>
@endsection

@push('script')
    <script>
        $(document).ready(function() {
            "use strict";

            const slugify = str =>
                `**${ str.toLowerCase().trim().replace( /[^\w\s-]/g, '' ).replace( /[\s_-]+/g, '-' ).replace( /^-+|-+$/g, '' ) }** `;
            /** @type {HTMLTemplateElement} */

            const repeaterInputTemplate = document.querySelector('#repeater-input-announcement');
            const addMorePlaceholder = document.querySelector('.add-more-placeholder');
            let currentInputGroups = document.querySelectorAll('.repeater-input-group');
            let lastInputsParent = [...currentInputGroups].at(-1);
            let lastInpusGroupId = lastInputsParent ? parseInt(lastInputsParent.getAttribute('data-inputs-id'),
                10) : 0;

            function formFillCheck() {
                const currentInputs = document.querySelectorAll(
                    '.announce_title');
                let anInputIsEmpty = false;
                currentInputs.forEach(input => {
                    const {
                        value
                    } = input;
                    if (!value || value.length === 0 || value.replace(/\s/g, '') === '') {
                        return anInputIsEmpty = true;
                    }
                });

                return anInputIsEmpty;
            }

            $(".add-more").click(function() {
                const button = this;
                if (formFillCheck()) {
                    return toastr.error('Please fill all fields in Input Group areas.');
                }
                const newInputsMarkup = repeaterInputTemplate.content.cloneNode(true);
                const newInputsWrapper = newInputsMarkup.firstElementChild;
                newInputsWrapper.dataset.inputsId = lastInpusGroupId + 1;
                addMorePlaceholder.before(newInputsMarkup);
                currentInputGroups = document.querySelectorAll('.repeater-input-group');
                lastInputsParent = [...currentInputGroups].at(-1);
                if (currentInputGroups.length > 1) {
                    document.querySelectorAll('.remove-inputs-group').forEach(el => el.removeAttribute(
                        'disabled'));
                }
                lastInpusGroupId++;
                const timeout = setTimeout(() => {
                    newInputsWrapper.querySelector('#title').focus();
                    clearTimeout(timeout);
                }, 100);
                return;
            });

            $("body").on("click", ".remove-inputs-group", function() {
                const button = $(this);
                const parent = button.closest('.repeater-input-group');
                const inputsId = parent.attr('data-inputs-id');

                $(`[data-inputs-id=${ inputsId }]`).remove();

                currentInputGroups = document.querySelectorAll('.repeater-input-group');
                lastInputsParent = [...currentInputGroups].at(-1);

                if (currentInputGroups.length > 1) {
                    document.querySelectorAll('.remove-inputs-group').forEach(el => el.removeAttribute(
                        'disabled'));
                } else {
                    document.querySelectorAll('.remove-inputs-group').forEach(el => el.setAttribute(
                        'disabled', true));
                }
            });

            $('#public_announcement_button').click(function() {
                if (formFillCheck()) {
                    return toastr.error('Please fill all fields in Input Group areas.');
                }

                document.getElementById("public_announcement_button").innerHTML = magicai_localize
                    .please_wait;
                document.getElementById("public_announcement_button").disabled = true;

                var formData = new FormData();
                $(".announce_active").each(function() {
                    formData.append('active[]', $(this).is(':checked'));
                });

                $(".announce_title").each(function() {
                    formData.append('title[]', $(this).val());
                });

                $(".announce_type").each(function() {
                    formData.append('type[]', $(this).val());
                });

                formData.append('_method', $('[name="_method"]').val());

                $.ajax({
                    type: "post",
                    url: '{{ $action }}',
                    data: formData,
                    contentType: false,
                    processData: false,
                    success: function(data) {
                        toastr.success('Saved Succesfully.');
                        document.getElementById("public_announcement_button").disabled = false;
                        document.getElementById("public_announcement_button").innerHTML =
                            "Save";
                        setTimeout(function() {
                            window.location.href = "/dashboard/admin/announcements";
                        }, 200);
                    },
                    error: function(data) {
                        var err = data.responseJSON.errors;
                        $.each(err, function(index, value) {
                            toastr.error(value);
                        });
                        document.getElementById("public_announcement_button").disabled = false;
                        document.getElementById("public_announcement_button").innerHTML =
                            "Save";
                    }
                });
                return false;
            })
        });
    </script>
@endpush
