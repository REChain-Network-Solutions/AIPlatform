## Что нового в SDK версии 3.x

* Новая версия SDK не поддерживает PHP ниже 8.0.
* Произошли изменения в моделях:
  * Добавлены атрибуты к свойствам.
  * Подготовка и валидация данных теперь осуществляется в базовом классе ```AbstractObject```.
  * Валидация реализована через собственную библиотеку ```yookassa-sdk-validator```.
  * Добавлена работа с коллекциями в свойствах с типом массив объектов.
    Интерфейс коллекции подразумевает, что в коллекции находятся объекты одного типа. Чтобы сменить тип объекта коллекции, необходимо сначала очистить коллекцию методом ```clear()```.
    Для добавления объекта в коллекцию необходимо вызвать метод ```add()```. Чтобы удалить объект по индексу, необходимо вызвать метод ```remove()```.
* Добавлены анализаторы кода.
* Классы распределены по папкам в соответствии с их назначением:

| **Папка**                          | Модель/интерфейс                        | **Новый namespace**                               | **Старый namespace**                      |
|------------------------------------|-----------------------------------------|---------------------------------------------------|-------------------------------------------|
| Deal                               | BaseDeal                                | YooKassa\Model\Deal                               | YooKassa\Model                            |
|                                    | DealInterface                           | YooKassa\Model\Deal                               | YooKassa\Model                            |
|                                    | SafeDeal                                | YooKassa\Model\Deal                               | YooKassa\Model                            |
| Notification                       | NotificationEventType                   | YooKassa\Model\Notification                       | YooKassa\Model                            |
|                                    | NotificationType                        | YooKassa\Model\Notification                       | YooKassa\Model                            |
| Payment/Confirmation               | AbstractConfirmation                    | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationCodeVerification            | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationEmbedded                    | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationExternal                    | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationFactory                     | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationMobileApplication           | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationQr                          | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
|                                    | ConfirmationRedirect                    | YooKassa\Model\Payment\Confirmation               | YooKassa\Model\Confirmation               |
| Payment/PaymentMethod/B2b/Sberbank | PayerBankDetails                        | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
|                                    | PayerBankDetailsInterface               | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
|                                    | VatDataInterface                        | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
|                                    | VatDataInterface                        | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
|                                    | VatDataRate                             | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
|                                    | VatDataType                             | YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank | YooKassa\Model\PaymentMethod\B2b\Sberbank |
| Payment/PaymentMethod/B2b          | AbstractPaymentMethod                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | BankCard                                | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | BankCardSource                          | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodCardType                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodAlfaBank                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodApplePay                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodB2bSberbank                | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodBankCard                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodCash                       | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodFactory                    | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodGooglePay                  | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodInstallments               | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodMobileBalance              | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodPsb                        | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodQiwi                       | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodSberbank                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodSbp                        | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodTinkoffBank                | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodWebmoney                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodWechat                     | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
|                                    | PaymentMethodYooMoney                   | YooKassa\Model\Payment\PaymentMethod              | YooKassa\Model\PaymentMethod              |
| Payment                            | AuthorizationDetailsInterface           | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | CancellationDetails                     | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | CancellationDetailsPartyCode            | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | CancellationDetailsReasonCode           | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | ConfirmationType                        | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | Payment                                 | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | PaymentInterface                        | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | PaymentMethodType                       | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | PaymentStatus                           | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | ReceiptRegistrationStatus               | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | Recipient                               | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | RecipientInterface                      | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | ThreeDSecure                            | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | Transfer                                | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | TransferInterface                       | YooKassa\Model\Payment                            | YooKassa\Model                            |
|                                    | TransferStatus                          | YooKassa\Model\Payment                            | YooKassa\Model                            |
| Payout                             | Payout                                  | YooKassa\Model\Payout                             | YooKassa\Model                            |
|                                    | DealInterface                           | YooKassa\Model\Payout                             | YooKassa\Model                            |
|                                    | PayoutStatus                            | YooKassa\Model\Payout                             | YooKassa\Model                            |
| Receipt                            | Receipt                                 | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptCustomer                         | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptCustomerInterface                | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptInterface                        | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptItem                             | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptItemInterface                    | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | ReceiptType                             | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | Settlement                              | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | SettlementInterface                     | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | Supplier                                | YooKassa\Model\Receipt                            | YooKassa\Model                            |
|                                    | SupplierInterface                       | YooKassa\Model\Receipt                            | YooKassa\Model                            |
| Refund                             | Refund                                  | YooKassa\Model\Refund                             | YooKassa\Model                            |
|                                    | RefundInterface                         | YooKassa\Model\Refund                             | YooKassa\Model                            |
|                                    | RefundStatus                            | YooKassa\Model\Refund                             | YooKassa\Model                            |
|                                    | Source                                  | YooKassa\Model\Refund                             | YooKassa\Model                            |
|                                    | SourceInterface                         | YooKassa\Model\Refund                             | YooKassa\Model                            |
| Payments/ConfirmationAttributes    | AbstractConfirmationAttributes          | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesCodeVerification  | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesEmbedded          | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesExternal          | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesFactory           | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesMobileApplication | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesQr                | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
|                                    | ConfirmationAttributesRedirect          | YooKassa\Request\Payments\ConfirmationAttributes  | YooKassa\Model\ConfirmationAttributes     |
| Payments/PaymentData               | AbstractPaymentData                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataAlfabank                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataApplePay                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataB2BSberbank                  | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataBankCard                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataBankCardCard                 | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataCash                         | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataFactory                      | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataGooglePay                    | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataInstallments                 | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataMobileBalance                | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataQiwi                         | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataSberbank                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataSbp                          | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataTinkoffBank                  | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataWebmoney                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataWechat                       | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
|                                    | PaymentDataYooMoney                     | YooKassa\Request\Payments\PaymentData             | YooKassa\Model\PaymentData                |
| Payments                           | AbstractPaymentRequest                  | YooKassa\Request\Payments                         | YooKassa\Common                           |
|                                    | AbstractPaymentRequestBuilder           | YooKassa\Request\Payments                         | YooKassa\Common                           |
|                                    | Airline                                 | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | AirlineInterface                        | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | CancelResponse                          | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | CreateCaptureRequest                    | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | CreateCaptureRequestBuilder             | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | CreateCaptureRequestInterface           | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | CreateCaptureRequestSerializer          | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | CreateCaptureResponse                   | YooKassa\Request\Payments                         | YooKassa\Request\Payments\Payment         |
|                                    | FraudData                               | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | Leg                                     | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | LegInterface                            | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | Locale                                  | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | Passenger                               | YooKassa\Request\Payments                         | YooKassa\Model                            |
|                                    | PassengerInterface                      | YooKassa\Request\Payments                         | YooKassa\Model                            |
