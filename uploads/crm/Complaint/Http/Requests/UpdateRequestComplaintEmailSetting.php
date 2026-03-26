<?php

namespace Modules\Complaint\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UpdateRequestComplaintEmailSetting extends FormRequest
{

    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {
        if (request('status')) {
            return [
                'mail_from_name' => 'required',
                'mail_from_email' => 'required',
                'mail_username' => 'required',
                'mail_password' => 'required',
                'imap_host' => 'required',
                'imap_port' => 'required',
                'imap_encryption' => 'required'
            ];
        }
        
        return [];
    }

}
