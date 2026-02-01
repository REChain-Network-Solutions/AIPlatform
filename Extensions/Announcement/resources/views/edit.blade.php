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
            @method('put')
            @csrf
            <x-form-step label="{{ __('Update announcement') }}" />

            <x-card
                class="relative"
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
                    :checked="$announcement->active"
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
                    :value="$announcement->title"
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
                        <option
                            value="{{ $type->value }}"
                            @selected($type->value == $announcement->type->value)
                        >
                            {{ $type->label() }}
                        </option>
                    @endforeach
                </x-forms.input>

            </x-card>

            <x-button
                id="public_announcement_button"
                type="button"
                form="public_announcement_form"
            >
                {{ __('Save') }}
            </x-button>

        </form>
    </div>
@endsection

@push('script')
    <script>
        $(document).ready(function() {
            "use strict";

            const slugify = str =>
                `**${ str.toLowerCase().trim().replace( /[^\w\s-]/g, '' ).replace( /[\s_-]+/g, '-' ).replace( /^-+|-+$/g, '' ) }** `;
            /** @type {HTMLTemplateElement} */

            function formFillCheck() {
                const title = document.querySelector('.announce_title');
                return (!title.value || title.value.length === 0 || title.value.replace(/\s/g, '') === '');
            }

            $('#public_announcement_button').click(function() {
                if (formFillCheck()) {
                    return toastr.error('Please fill all fields.');
                }

                document.getElementById("public_announcement_button").innerHTML = magicai_localize
                    .please_wait;
                document.getElementById("public_announcement_button").disabled = true;

                var formData = new FormData();
                formData.append('active', $("#active").is(':checked'));
                formData.append('title', $("#title").val());
                formData.append('type', $("#type").val());

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
