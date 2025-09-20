## Работа со способами оплаты

> Только для тех, кто использует [привязку на нулевую сумму](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/save-payment-method/save-without-payment) для сохранения банковских карт.

Способ оплаты — данные платежного средства пользователя. Например, данные банковской карты.

С помощью SDK вы можете создать способ оплаты — сохранить платежные данные в ЮKassa с привязкой к вашему магазину. Также через SDK вы можете получить актуальную информацию о созданном способе оплаты.

* [Запрос на создание способа оплаты](#Запрос-на-создание-способа-оплаты)
* [Запрос на создание способа оплаты через билдер](#Запрос-на-создание-способа-оплаты-через-билдер)
* [Получить информацию о способе оплаты](#Получить-информацию-о-способе-оплаты)

### Запрос на создание способа оплаты <a name="Запрос-на-создание-способа-оплаты"></a>

[Создание способа оплаты в документации](https://yookassa.ru/developers/api#create_payment_method)

Объект способа оплаты `SavePaymentMethod` содержит всю информацию о платежном средстве пользователя, актуальную на текущий момент времени. Он формируется при создании способа оплаты и приходит в ответ на любой запрос, связанный со способами оплаты.

Набор возвращаемых параметров зависит от статуса объекта (значение параметра `status`) и того, какие параметры вы передали в запросе на создание способа оплаты.

В ответ на запрос придет объект способа оплаты - `SavePaymentMethod` в актуальном статусе.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $response = $client->createPaymentMethod(
        [
            'confirmation' => [
                'enforce' => true,
                'type' => 'redirect',
                'return_url' => 'https://www.example.com/return_url',
                'locale' => 'ru_RU'
            ],
            'type' => 'bank_card',
            'card' => [
                'number' => '4100000000000002',
                'expiry_year' => '2025',
                'expiry_month' => '12',
                'cardholder' => 'Inna Ivanovna',
                'csc' => '123'
            ],
            'holder' => [
                'gateway_id' => '123456'
            ],
            'client_ip' => '127.0.0.1',
        ],
        uniqid('', true)
    );
    if ($response->getStatus() === SavePaymentMethodStatus::PENDING) {
        echo $response->getConfirmation()->getConfirmationUrl(); // Получаем ссылку на URL, на который необходимо перенаправить пользователя для подтверждения привязки
    }
} catch (\Exception $e) {
    var_dump($response);
}
```
---

### Запрос на создание способа оплаты через билдер <a name="Запрос-на-создание-способа-оплаты-через-билдер"></a>

[Создание способа оплаты в документации](https://yookassa.ru/developers/api#create_payment_method)

Билдер позволяет создать объект запроса на создание способа оплаты — `CreatePaymentMethodRequestBuilder` программным способом, через объекты.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $paymentMethodBuilder = \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest::builder();
    $paymentMethodBuilder
        ->setType(\YooKassa\Model\SavePaymentMethod\SavePaymentMethodType::BANK_CARD)
        ->setCard(new \YooKassa\Request\PaymentMethods\PaymentMethodCard([
            'number' => '4100000000000002',
            'expiry_year' => '2025',
            'expiry_month' => '12',
            'cardholder' => 'Ivan Ivanov',
            'csc' => '123',
        ]))
        ->setClientIp('127.0.0.1')
    ;

    // Создаем объект запроса
    $request = $paymentMethodBuilder->build();

    // Можно изменить данные, если нужно
    $request->setClientIp('127.0.0.2');

    $idempotenceKey = uniqid('', true);
    $response = $client->createPaymentMethod($request, $idempotenceKey);
    if ($response->getStatus() === SavePaymentMethodStatus::PENDING) {
        echo $response->getConfirmation()->getConfirmationUrl(); // Получаем ссылку на URL, на который необходимо перенаправить пользователя для подтверждения привязки
    }
} catch (Exception $e) {
    var_dump($e);
}
```
---

### Получить информацию о способе оплаты <a name="Получить-информацию-о-способе-оплаты"></a>

[Информация о способе оплаты в документации](https://yookassa.ru/developers/api#get_payment_method)

Запрос позволяет получить информацию о текущем состоянии способа оплаты по его уникальному идентификатору.

В ответ на запрос придет объект способа оплаты — `PaymentMethodResponse` — в актуальном статусе.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $paymentMethodId = '285c0ab7-0003-5000-9000-0e1166498fda';
    $response = $client->getPaymentMethodInfo($paymentMethodId);
    print_r($response->toArray());
    if ($response->getStatus() === SavePaymentMethodStatus::PENDING) {
        echo $response->getConfirmation()->getConfirmationUrl(); // Получаем ссылку на URL, на который необходимо перенаправить пользователя для подтверждения привязки
    }
} catch (Exception $e) {
    var_dump($e);
}
```
