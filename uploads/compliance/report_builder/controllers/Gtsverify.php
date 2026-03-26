<?php
if (!defined('BASEPATH')) {
    exit('No direct script access allowed');
}
require_once __DIR__ .'/../libraries/gtsslib.php';

/**
 * GTSSolution verify
 */
class Gtsverify extends AdminController{
    public function __construct(){
        parent::__construct();
    }

    /**
     * index 
     * @return void
     */
    public function index(){
        show_404();
    }

    /**
     * activate
     * @return json
     */
    public function activate(){
        $license_code = strip_tags(trim($_POST["purchase_key"]));
        $client_name = strip_tags(trim($_POST["username"])); 
        $api = new ReportBuilderLic();
        $activate_response = $api->activate_license($license_code, $client_name);
        $msg = '';
        if(empty($activate_response)){
          $msg = 'Server is unavailable.';
        }else{
          $msg = $activate_response['message'];
        }

        $res = array();
        $res['status'] = $activate_response['status'];
        $res['message'] = $msg;
        if ($res['status']) {
            $res['original_url']= $this->input->post('original_url');
        }
        echo json_encode($res);
    }    
}