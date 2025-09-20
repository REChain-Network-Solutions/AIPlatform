# [YooKassa API SDK](../home.md)

# Interface: InvoiceInterface
### Namespace: [\YooKassa\Model\Invoice](../namespaces/yookassa-model-invoice.md)
---
**Summary:**

Класс, представляющий модель Invoice.

**Description:**

Данные о счете.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
* [public MIN_LENGTH_ID](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#constant_MIN_LENGTH_ID)
* [public MAX_LENGTH_ID](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#constant_MAX_LENGTH_ID)
* [public MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#constant_MAX_LENGTH_DESCRIPTION)

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getCancellationDetails) |  | Возвращает cancellation_details. |
| public | [getCart()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getCart) |  | Возвращает cart. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getCreatedAt) |  | Возвращает created_at. |
| public | [getDeliveryMethod()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getDeliveryMethod) |  | Возвращает delivery_method. |
| public | [getDescription()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getDescription) |  | Возвращает description. |
| public | [getExpiresAt()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getExpiresAt) |  | Возвращает expires_at. |
| public | [getId()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getId) |  | Возвращает id. |
| public | [getMetadata()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPaymentDetails()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getPaymentDetails) |  | Возвращает payment_details. |
| public | [getStatus()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_getStatus) |  | Возвращает status. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setCancellationDetails) |  | Устанавливает cancellation_details. |
| public | [setCart()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setCart) |  | Устанавливает cart. |
| public | [setCreatedAt()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setCreatedAt) |  | Устанавливает created_at. |
| public | [setDeliveryMethod()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setDeliveryMethod) |  | Устанавливает delivery_method. |
| public | [setDescription()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setDescription) |  | Устанавливает description. |
| public | [setExpiresAt()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setExpiresAt) |  | Устанавливает expires_at. |
| public | [setId()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setId) |  | Устанавливает id. |
| public | [setMetadata()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPaymentDetails()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setPaymentDetails) |  | Устанавливает payment_details. |
| public | [setStatus()](../classes/YooKassa-Model-Invoice-InvoiceInterface.md#method_setStatus) |  | Устанавливает status. |

---
### Details
* File: [lib/Model/Invoice/InvoiceInterface.php](../../lib/Model/Invoice/InvoiceInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор счета в ЮKassa. |
| property |  | Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`. |
| property |  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |
| property |  | Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`. |
| property |  | Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`. |
| property |  | Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation). |
| property |  | Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation). |
| property |  | Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z` |
| property |  | Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z` |
| property |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`. |
| property |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`. |
| property |  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |
| property |  | Комментарий к статусу `canceled`: кто отменил счет и по какой причине. |
| property |  | Комментарий к статусу `canceled`: кто отменил счет и по какой причине. |
| property |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8./ |

---
## Constants
<a name="constant_MIN_LENGTH_ID" class="anchor"></a>
###### MIN_LENGTH_ID
```php
MIN_LENGTH_ID = 39 : int
```


<a name="constant_MAX_LENGTH_ID" class="anchor"></a>
###### MAX_LENGTH_ID
```php
MAX_LENGTH_ID = 39 : int
```


<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
```php
MAX_LENGTH_DESCRIPTION = 128 : int
```



---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает id.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** string|null - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает id.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор счета в ЮKassa. |

**Returns:** self - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает status.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** string|null - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает status.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  |  |

**Returns:** self - 


<a name="method_getCart" class="anchor"></a>
#### public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null

```php
public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null
```

**Summary**

Возвращает cart.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null - 


<a name="method_setCart" class="anchor"></a>
#### public setCart() : self

```php
public setCart(\YooKassa\Common\ListObjectInterface|array|null $cart = null) : self
```

**Summary**

Устанавливает cart.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | cart  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |

**Returns:** self - 


<a name="method_getDeliveryMethod" class="anchor"></a>
#### public getDeliveryMethod() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null

```php
public getDeliveryMethod() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null
```

**Summary**

Возвращает delivery_method.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null - 


<a name="method_setDeliveryMethod" class="anchor"></a>
#### public setDeliveryMethod() : self

```php
public setDeliveryMethod(\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|array|null $delivery_method = null) : self
```

**Summary**

Устанавливает delivery_method.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod OR array OR null</code> | delivery_method  |  |

**Returns:** self - 


<a name="method_getPaymentDetails" class="anchor"></a>
#### public getPaymentDetails() : \YooKassa\Model\Invoice\PaymentDetails|null

```php
public getPaymentDetails() : \YooKassa\Model\Invoice\PaymentDetails|null
```

**Summary**

Возвращает payment_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \YooKassa\Model\Invoice\PaymentDetails|null - Данные о платеже по выставленному счету


<a name="method_setPaymentDetails" class="anchor"></a>
#### public setPaymentDetails() : self

```php
public setPaymentDetails(\YooKassa\Model\Invoice\PaymentDetails|array|null $payment_details = null) : self
```

**Summary**

Устанавливает payment_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\PaymentDetails OR array OR null</code> | payment_details  | Данные о платеже по выставленному счету |

**Returns:** self - 


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \DateTime|null - Дата и время создания счета на оплату


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at) : self
```

**Summary**

Устанавливает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Дата и время создания счета на оплату |

**Returns:** self - 


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает expires_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \DateTime|null - Срок действия счета


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : self
```

**Summary**

Устанавливает expires_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Срок действия счета |

**Returns:** self - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** string|null - Описание выставленного счета


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |

**Returns:** self - 


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null

```php
public getCancellationDetails() : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null
```

**Summary**

Возвращает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \YooKassa\Model\Invoice\InvoiceCancellationDetails|null - Комментарий к статусу `canceled`


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(\YooKassa\Model\Invoice\InvoiceCancellationDetails|array|null $cancellation_details = null) : self
```

**Summary**

Устанавливает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\InvoiceCancellationDetails OR array OR null</code> | cancellation_details  | Комментарий к статусу `canceled` |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(string|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** self - 




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