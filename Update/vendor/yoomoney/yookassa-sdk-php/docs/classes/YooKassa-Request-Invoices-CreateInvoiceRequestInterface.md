# [YooKassa API SDK](../home.md)

# Interface: CreateInvoiceRequestInterface
### Namespace: [\YooKassa\Request\Invoices](../namespaces/yookassa-request-invoices.md)
---
**Summary:**

Класс, представляющий модель CreateInvoiceRequest.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [getCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getCart) |  | Возвращает cart. |
| public | [getDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getDeliveryMethodData) |  | Возвращает delivery_method_data. |
| public | [getDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getDescription) |  | Возвращает description. |
| public | [getExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getExpiresAt) |  | Возвращает expires_at. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getLocale) |  | Возвращает locale. |
| public | [getMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_getPaymentData) |  | Возвращает payment_data. |
| public | [setCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setCart) |  | Устанавливает cart. |
| public | [setDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setDeliveryMethodData) |  | Устанавливает delivery_method_data. |
| public | [setDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setDescription) |  | Устанавливает description. |
| public | [setExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setExpiresAt) |  | Устанавливает expires_at. |
| public | [setLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setLocale) |  | Устанавливает locale. |
| public | [setMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md#method_setPaymentData) |  | Устанавливает payment_data. |
| public | [validate()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_validate) |  | Валидирует текущий запрос, проверяет все ли нужные свойства установлены. |

---
### Details
* File: [lib/Request/Invoices/CreateInvoiceRequestInterface.php](../../lib/Request/Invoices/CreateInvoiceRequestInterface.php)
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
| property |  | Данные для проведения платежа по выставленному счету. |
| property |  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |
| property |  | Данные о способе доставки счета пользователю. Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом. |
| property |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z` |
| property |  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен. |
| property |  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |
| property |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

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


<a name="method_getPaymentData" class="anchor"></a>
#### public getPaymentData() : \YooKassa\Request\Invoices\PaymentData|null

```php
public getPaymentData() : \YooKassa\Request\Invoices\PaymentData|null
```

**Summary**

Возвращает payment_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** \YooKassa\Request\Invoices\PaymentData|null - Данные для проведения платежа по выставленному счету


<a name="method_setPaymentData" class="anchor"></a>
#### public setPaymentData() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setPaymentData(\YooKassa\Request\Invoices\PaymentData|array|null $payment_data = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает payment_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\PaymentData OR array OR null</code> | payment_data  | Данные для проведения платежа по выставленному счету |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getCart" class="anchor"></a>
#### public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface

```php
public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает cart.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface - Корзина заказа — список товаров или услуг


<a name="method_setCart" class="anchor"></a>
#### public setCart() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setCart(\YooKassa\Common\ListObjectInterface|array|null $cart = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает cart.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | cart  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getDeliveryMethodData" class="anchor"></a>
#### public getDeliveryMethodData() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null

```php
public getDeliveryMethodData() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null
```

**Summary**

Возвращает delivery_method_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null - Данные о способе доставки счета пользователю


<a name="method_setDeliveryMethodData" class="anchor"></a>
#### public setDeliveryMethodData() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setDeliveryMethodData(\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|array|null $delivery_method_data = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает delivery_method_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData OR array OR null</code> | delivery_method_data  | Данные о способе доставки счета пользователю |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает expires_at.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** \DateTime|null - Срок действия счета


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает expires_at.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z` |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getLocale" class="anchor"></a>
#### public getLocale() : string|null

```php
public getLocale() : string|null
```

**Summary**

Возвращает locale.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** string|null - Язык интерфейса, писем и смс, которые будет видеть или получать пользователь


<a name="method_setLocale" class="anchor"></a>
#### public setLocale() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setLocale(string|null $locale = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает locale.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | locale  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** string|null - Описание выставленного счета


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setDescription(string|null $description = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface - 




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