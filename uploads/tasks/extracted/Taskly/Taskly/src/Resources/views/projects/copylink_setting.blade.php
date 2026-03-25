@php
    $password = base64_decode($project->password);
    $setting = [];
    if ($project->copylinksetting) {
        $setting = json_decode($project->copylinksetting);
    }
@endphp
{{ Form::open(['route' => ['project.setting.save', $project->id], 'method' => 'POST']) }}
<div class="modal-body">
    <div class="table-responsive">
        <table class="table mb-0">
            <thead class="thead-light">
                <tr>
                    <th> {{ __('Name') }}</th>
                    <th class="text-right"> {{ __('Off/On') }}</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{{ __('Basic details') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="basic_details" class="form-check-input"
                                @if (isset($setting->basic_details) && $setting->basic_details == 'on') checked="checked" @endif id="copy_link_1"
                                value="on">
                            <label class="custom-control-label" for="copy_link_1"></label>
                        </div>
                    </td>
                </tr>
                @if ($project->type == 'project')
                    <tr>
                        <td>{{ __('Member') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="member" class="form-check-input"
                                    @if (isset($setting->member) && $setting->member == 'on') checked="checked" @endif id="copy_link_2"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_2"></label>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td>{{ __('Client') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="client" class="form-check-input"
                                    @if (isset($setting->client) && $setting->client == 'on') checked="checked" @endif id="copy_link_21"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_21"></label>
                            </div>
                        </td>
                    </tr>
                    @if (module_is_active('Account'))
                        <tr>
                            <td>{{ __('Vendor') }}</td>
                            <td class="action text-right">
                                <div class="form-check form-switch">
                                    <input type="checkbox" name="vendor" class="form-check-input"
                                        @if (isset($setting->vendor) && $setting->vendor == 'on') checked="checked" @endif id="copy_link_22"
                                        value="on">
                                    <label class="custom-control-label" for="copy_link_22"></label>
                                </div>
                            </td>
                        </tr>
                    @endif
                @endif
                <tr>
                    <td>{{ __('Milestone') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="milestone" class="form-check-input"
                                @if (isset($setting->milestone) && $setting->milestone == 'on') checked="checked" @endif id="copy_link_3"
                                value="on">
                            <label class="custom-control-label" for="copy_link_3"></label>
                        </div>
                    </td>
                </tr>
                @if ($project->type == 'project')
                    <tr>
                        <td>{{ __('Activity') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="activity" class="form-check-input"
                                    @if (isset($setting->activity) && $setting->activity == 'on') checked="checked" @endif id="copy_link_4"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_4"></label>
                            </div>
                        </td>
                    </tr>
                @endif
                <tr>
                    <td>{{ __('Files') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="attachment" class="form-check-input"
                                @if (isset($setting->attachment) && $setting->attachment == 'on') checked="checked" @endif id="copy_link_5"
                                value="on">
                            <label class="custom-control-label" for="copy_link_5"></label>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>{{ __('Task') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="task" class="form-check-input" id="copy_link_7"
                                @if (isset($setting->task) && $setting->task == 'on') checked="checked" @endif value="on">
                            <label class="custom-control-label" for="copy_link_7"></label>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>{{ __('Bug Report') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="bug_report" class="form-check-input"
                                @if (isset($setting->bug_report) && $setting->bug_report == 'on') checked="checked" @endif id="copy_link_6"
                                value="on">
                            <label class="custom-control-label" for="copy_link_6"></label>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>{{ __('Invoice') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="invoice" class="form-check-input" id="copy_link_7_1"
                                @if (isset($setting->invoice) && $setting->invoice == 'on') checked="checked" @endif value="on">
                            <label class="custom-control-label" for="copy_link_7_1"></label>
                        </div>
                    </td>
                </tr>
                @if (module_is_active('Account'))
                    <tr>
                        <td>{{ __('Bill') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="bill" class="form-check-input" id="copy_link_7_1_2"
                                    @if (isset($setting->bill) && $setting->bill == 'on') checked="checked" @endif value="on">
                                <label class="custom-control-label" for="copy_link_7_1_2"></label>
                            </div>
                        </td>
                    </tr>
                @endif

                @if (module_is_active('Timesheet'))
                    <tr>
                        <td>{{ __('Timesheet') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="timesheet" class="form-check-input"
                                    @if (isset($setting->timesheet) && $setting->timesheet == 'on') checked="checked" @endif id="copy_link_9"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_9"></label>
                            </div>
                        </td>
                    </tr>
                @endif
                @if (module_is_active('Documents'))
                    <tr>
                        <td>{{ __('Documents') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="documents" class="form-check-input"
                                    @if (isset($setting->documents) && $setting->documents == 'on') checked="checked" @endif id="copy_link_9"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_9"></label>
                            </div>
                        </td>
                    </tr>
                @endif
                <tr>
                    <td>{{ __('Progress') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="progress" class="form-check-input"
                                @if (isset($setting->progress) && $setting->progress == 'on') checked="checked" @endif id="copy_link_10"
                                value="on">
                            <label class="custom-control-label" for="copy_link_10"></label>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>{{ __('Password Protected') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="password_protected"
                                class="form-check-input password_protect" id="password_protected"
                                @if (isset($setting->password_protected) && $setting->password_protected == 'on') checked="checked" @endif value="on">
                            <label class="custom-control-label" for="password_protected"></label>
                        </div>
                    </td>
                <tr class="passwords">
                    <td>
                        <div class="action input-group input-group-merge  text-left ">
                            <input type="password" value="{{ $password }}"
                                class=" form-control @error('password') is-invalid @enderror" name="password"
                                autocomplete="new-password" id="password"
                                placeholder="{{ __('Enter Your Password') }}">
                            <div class="input-group-append">
                                <span class="input-group-text py-3">
                                    <a href="#" data-toggle="password-text" data-target="#password">
                                        <i class="fas fa-eye-slash" id="togglePassword"></i>
                                    </a>
                                </span>
                            </div>
                        </div>
                    </td>
                </tr>
                </tr>
                @if (module_is_active('Retainer'))
                    <tr>
                        <td>{{ __('Retainer') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="retainer" class="form-check-input"
                                    @if (isset($setting->retainer) && $setting->retainer == 'on') checked="checked" @endif id="copy_link_10"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_10"></label>
                            </div>
                        </td>
                    </tr>
                @endif
                <tr>
                    <td>{{ __('Proposal') }}</td>
                    <td class="action text-right">
                        <div class="form-check form-switch">
                            <input type="checkbox" name="proposal" class="form-check-input"
                                @if (isset($setting->proposal) && $setting->proposal == 'on') checked="checked" @endif id="copy_link_12"
                                value="on">
                            <label class="custom-control-label" for="copy_link_12"></label>
                        </div>
                    </td>
                </tr>
                @if (module_is_active('Procurement'))
                    <tr>
                        <td>{{ __('Procurement') }}</td>
                        <td class="action text-right">
                            <div class="form-check form-switch">
                                <input type="checkbox" name="procurement" class="form-check-input"
                                    @if (isset($setting->procurement) && $setting->procurement == 'on') checked="checked" @endif id="copy_link_14"
                                    value="on">
                                <label class="custom-control-label" for="copy_link_14"></label>
                            </div>
                        </td>
                    </tr>
                @endif
            </tbody>
        </table>
    </div>
</div>
<div class="modal-footer">
    {{ Form::button(__('Save'), ['type' => 'submit', 'class' => 'btn btn-primary']) }}
</div>
{{ Form::close() }}

<script>
    $(document).ready(function() {
        if ($('.password_protect').is(':checked')) {
            $('.passwords').show();
        } else {
            $('.passwords').hide();
        }
        $('#password_protected').on('change', function() {
            if ($('.password_protect').is(':checked')) {
                $('.passwords').show();
            } else {
                $('.passwords').hide();
            }
        });
    });
    $(document).on('change', '#password_protected', function() {
        if ($(this).is(':checked')) {
            $('.passwords').removeClass('password_protect');
            $('.passwords').attr("required", true);
        } else {
            $('.passwords').addClass('password_protect');
            $('.passwords').val(null);
            $('.passwords').removeAttr("required");
        }
    });
</script>
<script>
    const togglePassword = document.querySelector("#togglePassword");
    const password = document.querySelector("#password");

    togglePassword.addEventListener("click", function() {
        // toggle the type attribute
        const type = password.getAttribute("type") === "password" ? "text" : "password";
        password.setAttribute("type", type);

        this.classList.toggle("fa-eye");
        this.classList.toggle("fa-eye-slash");
    });
</script>
