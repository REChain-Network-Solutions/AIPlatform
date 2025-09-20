# [YooKassa API SDK](../home.md)

# Interface: CreatePaymentMethodRequestInterface
### Namespace: [\YooKassa\Request\PaymentMethods](../namespaces/yookassa-request-paymentmethods.md)
---
**Summary:**

Класс, представляющий модель CreatePaymentMethodRequest.

**Description:**

Данные для проверки и сохранения способа оплаты.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [getCard()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_getCard) |  | Возвращает card. |
| public | [getClientIp()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_getClientIp) |  | Возвращает client_ip. |
| public | [getConfirmation()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_getConfirmation) |  | Возвращает confirmation. |
| public | [getHolder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_getHolder) |  | Возвращает holder. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_getType) |  | Возвращает type. |
| public | [setCard()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_setCard) |  | Устанавливает card. |
| public | [setClientIp()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_setClientIp) |  | Устанавливает client_ip. |
| public | [setConfirmation()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_setConfirmation) |  | Устанавливает confirmation. |
| public | [setHolder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_setHolder) |  | Устанавливает holder. |
| public | [setType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md#method_setType) |  | Устанавливает type. |
| public | [validate()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_validate) |  | Валидирует текущий запрос, проверяет все ли нужные свойства установлены. |

---
### Details
* File: [lib/Request/PaymentMethods/CreatePaymentMethodRequestInterface.php](../../lib/Request/PaymentMethods/CreatePaymentMethodRequestInterface.php)
* Package: \YooKassa\Model
* Parents:
  * [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |
| property |  | Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне). |
| property |  | Данные магазина, для которого сохраняется способ оплаты. |
| property |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| property |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| property |  | Данные, необходимые для инициирования сценария подтверждения привязки. |

---
## Methods
<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Валидирует текущий запрос, проверяет все ли нужные свойства установлены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** bool - True если запрос валиден, false если нет


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** void - 


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

**Returns:** string|null - 


<a name="method_setType" class="anchor"></a>
#### public setType() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
public setType(string|null $type = null) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - 


<a name="method_getCard" class="anchor"></a>
#### public getCard() : \YooKassa\Request\PaymentMethods\PaymentMethodCard|null

```php
public getCard() : \YooKassa\Request\PaymentMethods\PaymentMethodCard|null
```

**Summary**

Возвращает card.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

**Returns:** \YooKassa\Request\PaymentMethods\PaymentMethodCard|null - 


<a name="method_setCard" class="anchor"></a>
#### public setCard() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
public setCard(\YooKassa\Request\PaymentMethods\PaymentMethodCard|array|null $card) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Устанавливает card.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodCard OR array OR null</code> | card  | Данные для проверки и сохранения банковской карты. |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - 


<a name="method_getHolder" class="anchor"></a>
#### public getHolder() : \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null

```php
public getHolder() : \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null
```

**Summary**

Возвращает holder.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

**Returns:** \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null - 


<a name="method_setHolder" class="anchor"></a>
#### public setHolder() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
public setHolder(\YooKassa\Request\PaymentMethods\PaymentMethodHolder|array|null $holder = null) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Устанавливает holder.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodHolder OR array OR null</code> | holder  | Данные магазина, для которого сохраняется способ оплаты. |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - 


<a name="method_getClientIp" class="anchor"></a>
#### public getClientIp() : string|null

```php
public getClientIp() : string|null
```

**Summary**

Возвращает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

**Returns:** string|null - 


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
public setClientIp(string|null $client_ip = null) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Устанавливает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | client_ip  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - 


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null

```php
public getConfirmation() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null
```

**Summary**

Возвращает confirmation.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

**Returns:** \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null - 


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest

```php
public setConfirmation(\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|array|null $confirmation = null) : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
```

**Summary**

Устанавливает confirmation.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation OR array OR null</code> | confirmation  | Данные, необходимые для инициирования сценария подтверждения привязки. |

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest - 




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