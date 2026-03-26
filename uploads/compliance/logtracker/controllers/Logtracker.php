<?php

defined('BASEPATH') or exit('No direct script access allowed');

class Logtracker extends AdminController
{
    public function __construct()
    {
        parent::__construct();
    }

    public function index()
    {
        if (!has_permission('logtracker', '', 'view')) {
            access_denied();
        }

        $viewData['title'] = _l('logtracker_dashboard');
        $viewData['summary'] = getLogData("logDate");

        $this->load->view('manage', $viewData);
    }

    public function view($date)
    {
        if (!has_permission('logtracker', '', 'view')) {
            access_denied();
        }

        $viewData['title'] = _l('log') . '[' . $date . ']';
        $logData = getLogData("logDate");

        $viewData['log_data'] = $logData['data'][$date];
        if (empty($viewData['log_data'])) {
            set_alert('danger', _l('no_details_found'));
            redirect(admin_url('logtracker'));
        }
        $viewData['selected_date'] = $date;

        $this->load->view('view', $viewData);
    }

    public function get_table_data($tableName, $date = "")
    {
        if (!$this->input->is_ajax_request()) {
            ajax_access_denied();
        }

        $this->app->get_table_data(module_views_path(LOGTRACKER_MODULE, 'tables/' . $tableName), ['date' => $date]);
    }

    public function createZip($filePath)
    {
        $zip = new ZipArchive();
        $zipFilePath = $filePath . '.zip';

        if ($zip->open($zipFilePath, ZipArchive::CREATE | ZipArchive::OVERWRITE)) {
            $zip->addFile($filePath, basename($filePath));
            $zip->close();
            return $zipFilePath;
        } else {
            return false;
        }
    }

    public function downloadLogFile($fileName, $zip = false)
    {
        if (!has_permission('logtracker', '', 'download')) {
            access_denied();
        }

        $folderPath = !empty($folderPath) ? $folderPath : APPPATH . '/logs';
        $filePath = $folderPath . '/' . $fileName;

        if (file_exists($filePath . '.php')) {
            if ($zip) {
                $zip = new ZipArchive();
                $zipFilePath = $filePath . '.zip';

                if ($zip->open($zipFilePath, ZipArchive::CREATE | ZipArchive::OVERWRITE)) {
                    $zip->addFile($filePath . '.php', basename($filePath . '.txt'));
                    $zip->close();

                    header('Content-Type: application/zip');
                    header('Content-Disposition: attachment; filename="' . basename($zipFilePath) . '"');
                    header('Content-Length: ' . filesize($zipFilePath));
                    readfile($zipFilePath);

                    unlink($zipFilePath);
                    exit;
                }
            }

            header('Content-Description: File Transfer');
            header('Content-Type: application/octet-stream');
            header('Content-Disposition: attachment; filename="' . basename($filePath . '.txt') . '"');
            header('Expires: 0');
            header('Cache-Control: must-revalidate');
            header('Pragma: public');
            header('Content-Length: ' . filesize($filePath . '.php'));
            readfile($filePath . '.php');
            exit;
        }
    }

    public function deleteLogFileUsingAjax($fileName)
    {
        if (!$this->input->is_ajax_request()) {
            ajax_access_denied();
        }

        $folderPath = !empty($folderPath) ? $folderPath : APPPATH . '/logs';
        $filePath = $folderPath . '/' . $fileName . '.php';

        if (file_exists($filePath)) {
            $fileDeleted = unlink($filePath);
            echo json_encode(['type' => 'danger', 'message' => $fileDeleted ? _l('log_file_deleted') : _l('something_went_wrong')]);
        }
    }

    public function deleteLogFile($fileName)
    {
        if (!has_permission('logtracker', '', 'delete')) {
            access_denied();
        }

        $folderPath = !empty($folderPath) ? $folderPath : APPPATH . '/logs';
        $filePath = $folderPath . '/' . $fileName . '.php';

        if (file_exists($filePath)) {
            $fileDeleted = unlink($filePath);
            set_alert('danger', $fileDeleted ? _l('log_file_deleted') : _l('something_went_wrong'));
            redirect(admin_url('logtracker'));
        }
    }

    public function sendErroLogMail()
    {
        $postData = $this->input->post();

        $sendMail = send_mail_template('error_log_information', LOGTRACKER_MODULE, $postData['email_to'], $postData['error_level'], $postData['error_time'], $postData['error_message']);

        echo json_encode([
            'type' => $sendMail ? 'success' : 'danger',
            'message' => $sendMail ? _l('mail_sent_success') : _l('mail_was_not_sent')
        ]);
    }
}
