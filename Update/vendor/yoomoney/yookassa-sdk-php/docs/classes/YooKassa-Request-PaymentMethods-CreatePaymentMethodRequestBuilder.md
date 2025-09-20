# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder
### Namespace: [\YooKassa\Request\PaymentMethods](../namespaces/yookassa-request-paymentmethods.md)
---
**Summary:**

Класс, представляющий модель CreatePaymentMethodRequestBuilder.

**Description:**

Класс билдера объекта запроса на создание платежа, передаваемого в методы клиента API.

---
### Examples
Пример использования билдера

```php
// Создание запроса на создание способа оплаты через билдер
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
    ;

    // Создаем объект запроса
    $request = $paymentMethodBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createPaymentMethod($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}


```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$currentObject](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [build()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setCard()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_setCard) |  | Устанавливает card. |
| public | [setClientIp()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_setClientIp) |  | Устанавливает client_ip. |
| public | [setConfirmation()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_setConfirmation) |  | Устанавливает confirmation. |
| public | [setHolder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_setHolder) |  | Устанавливает holder. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_setType) |  | Устанавливает type. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/PaymentMethods/CreatePaymentMethodRequestBuilder.php](../../lib/Request/PaymentMethods/CreatePaymentMethodRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

**Returns:** mixed - 


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface|\YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса


<a name="method_setCard" class="anchor"></a>
#### public setCard() : self

```php
public setCard(\YooKassa\Request\PaymentMethods\PaymentMethodCard|array|null $value = null) : self
```

**Summary**

Устанавливает card.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodCard OR array OR null</code> | value  | Данные для проверки и сохранения банковской карты. |

**Returns:** self - 


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : self

```php
public setClientIp(string|null $value = null) : self
```

**Summary**

Устанавливает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |

**Returns:** self - 


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : self

```php
public setConfirmation(\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|array|null $value = null) : self
```

**Summary**

Устанавливает confirmation.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation OR array OR null</code> | value  | Данные, необходимые для инициирования сценария подтверждения привязки. |

**Returns:** self - 


<a name="method_setHolder" class="anchor"></a>
#### public setHolder() : self

```php
public setHolder(\YooKassa\Request\PaymentMethods\PaymentMethodHolder|array|null $value = null) : self
```

**Summary**

Устанавливает holder.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodHolder OR array OR null</code> | value  | Данные магазина, для которого сохраняется способ оплаты. |

**Returns:** self - 


<a name="method_setOptions" class="anchor"></a>
#### public setOptions() : \YooKassa\Common\AbstractRequestBuilder

```php
public setOptions(iterable|null $options) : \YooKassa\Common\AbstractRequestBuilder
```

**Summary**

Устанавливает свойства запроса из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable OR null</code> | options  | Массив свойств запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \InvalidArgumentException | Выбрасывается если аргумент не массив и не итерируемый объект |
| \YooKassa\Common\Exceptions\InvalidPropertyException | Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек |

**Returns:** \YooKassa\Common\AbstractRequestBuilder - Инстанс текущего билдера запросов


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $value = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |

**Returns:** self - 


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
protected initCurrentObject() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestBuilder.md)

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - Инстанс собираемого объекта запроса к API



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney