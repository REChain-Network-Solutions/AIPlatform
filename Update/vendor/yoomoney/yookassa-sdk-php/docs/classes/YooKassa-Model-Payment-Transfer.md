# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\Transfer
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Класс, представляющий модель Transfer.

**Description:**

Данные о распределении денег — сколько и в какой магазин нужно перевести.
Присутствует, если вы используете [Сплитование платежей](/developers/solutions-for-platforms/split-payments/basics).

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Payment-Transfer.md#constant_MAX_LENGTH_DESCRIPTION) |  | Максимальная длина строки описания транзакции |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account_id](../classes/YooKassa-Model-Payment-Transfer.md#property_account_id) |  | Идентификатор магазина, в пользу которого вы принимаете оплату |
| public | [$accountId](../classes/YooKassa-Model-Payment-Transfer.md#property_accountId) |  | Идентификатор магазина, в пользу которого вы принимаете оплату |
| public | [$amount](../classes/YooKassa-Model-Payment-Transfer.md#property_amount) |  | Сумма, которую необходимо перечислить магазину |
| public | [$connected_account_id](../classes/YooKassa-Model-Payment-Transfer.md#property_connected_account_id) |  | Идентификатор продавца, подключенного к вашей платформе |
| public | [$connectedAccountId](../classes/YooKassa-Model-Payment-Transfer.md#property_connectedAccountId) |  | Идентификатор продавца, подключенного к вашей платформе |
| public | [$description](../classes/YooKassa-Model-Payment-Transfer.md#property_description) |  | Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72») |
| public | [$metadata](../classes/YooKassa-Model-Payment-Transfer.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа) |
| public | [$platform_fee_amount](../classes/YooKassa-Model-Payment-Transfer.md#property_platform_fee_amount) |  | Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу |
| public | [$platformFeeAmount](../classes/YooKassa-Model-Payment-Transfer.md#property_platformFeeAmount) |  | Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу |
| public | [$release_funds](../classes/YooKassa-Model-Payment-Transfer.md#property_release_funds) |  | Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать |
| public | [$releaseFunds](../classes/YooKassa-Model-Payment-Transfer.md#property_releaseFunds) |  | Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать |
| public | [$status](../classes/YooKassa-Model-Payment-Transfer.md#property_status) |  | Статус распределения денег между магазинами. Возможные значения: `pending`, `waiting_for_capture`, `succeeded`, `canceled` |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAccountId()](../classes/YooKassa-Model-Payment-Transfer.md#method_getAccountId) |  | Возвращает account_id. |
| public | [getAmount()](../classes/YooKassa-Model-Payment-Transfer.md#method_getAmount) |  | Возвращает amount. |
| public | [getConnectedAccountId()](../classes/YooKassa-Model-Payment-Transfer.md#method_getConnectedAccountId) |  | Возвращает идентификатор продавца. |
| public | [getDescription()](../classes/YooKassa-Model-Payment-Transfer.md#method_getDescription) |  | Возвращает description. |
| public | [getMetadata()](../classes/YooKassa-Model-Payment-Transfer.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPlatformFeeAmount()](../classes/YooKassa-Model-Payment-Transfer.md#method_getPlatformFeeAmount) |  | Возвращает platform_fee_amount. |
| public | [getReleaseFunds()](../classes/YooKassa-Model-Payment-Transfer.md#method_getReleaseFunds) |  | Возвращает порядок перевода денег продавцам. |
| public | [getStatus()](../classes/YooKassa-Model-Payment-Transfer.md#method_getStatus) |  | Возвращает status. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccountId()](../classes/YooKassa-Model-Payment-Transfer.md#method_setAccountId) |  | Устанавливает account_id. |
| public | [setAmount()](../classes/YooKassa-Model-Payment-Transfer.md#method_setAmount) |  | Устанавливает amount. |
| public | [setConnectedAccountId()](../classes/YooKassa-Model-Payment-Transfer.md#method_setConnectedAccountId) |  | Устанавливает идентификатор продавца. |
| public | [setDescription()](../classes/YooKassa-Model-Payment-Transfer.md#method_setDescription) |  | Устанавливает description. |
| public | [setMetadata()](../classes/YooKassa-Model-Payment-Transfer.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPlatformFeeAmount()](../classes/YooKassa-Model-Payment-Transfer.md#method_setPlatformFeeAmount) |  | Устанавливает platform_fee_amount. |
| public | [setReleaseFunds()](../classes/YooKassa-Model-Payment-Transfer.md#method_setReleaseFunds) |  | Устанавливает порядок перевода денег продавцам. |
| public | [setStatus()](../classes/YooKassa-Model-Payment-Transfer.md#method_setStatus) |  | Устанавливает status. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payment/Transfer.php](../../lib/Model/Payment/Transfer.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payment\Transfer
* Implements:
  * [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
Максимальная длина строки описания транзакции

```php
MAX_LENGTH_DESCRIPTION = 128
```



---
## Properties
<a name="property_account_id"></a>
#### public $account_id : string
---
***Description***

Идентификатор магазина, в пользу которого вы принимаете оплату

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_accountId"></a>
#### public $accountId : string
---
***Description***

Идентификатор магазина, в пользу которого вы принимаете оплату

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма, которую необходимо перечислить магазину

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_connected_account_id"></a>
#### public $connected_account_id : string
---
***Description***

Идентификатор продавца, подключенного к вашей платформе

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_connectedAccountId"></a>
#### public $connectedAccountId : string
---
***Description***

Идентификатор продавца, подключенного к вашей платформе

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72»)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа)

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_platform_fee_amount"></a>
#### public $platform_fee_amount : \YooKassa\Model\AmountInterface
---
***Description***

Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_platformFeeAmount"></a>
#### public $platformFeeAmount : \YooKassa\Model\AmountInterface
---
***Description***

Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_release_funds"></a>
#### public $release_funds : bool
---
***Description***

Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_releaseFunds"></a>
#### public $releaseFunds : bool
---
***Description***

Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус распределения денег между магазинами. Возможные значения: `pending`, `waiting_for_capture`, `succeeded`, `canceled`

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : string|null

```php
public getAccountId() : string|null
```

**Summary**

Возвращает account_id.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** string|null - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает amount.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_getConnectedAccountId" class="anchor"></a>
#### public getConnectedAccountId() : string|null

```php
public getConnectedAccountId() : string|null
```

**Summary**

Возвращает идентификатор продавца.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** string|null - Идентификатор продавца


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** string|null - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** \YooKassa\Model\Metadata|null - 


<a name="method_getPlatformFeeAmount" class="anchor"></a>
#### public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null

```php
public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает platform_fee_amount.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_getReleaseFunds" class="anchor"></a>
#### public getReleaseFunds() : bool|null

```php
public getReleaseFunds() : bool|null
```

**Summary**

Возвращает порядок перевода денег продавцам.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** bool|null - Порядок перевода денег продавцам


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает status.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setAccountId" class="anchor"></a>
#### public setAccountId() : self

```php
public setAccountId(string|null $value = null) : self
```

**Summary**

Устанавливает account_id.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  |  |

**Returns:** self - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $value = null) : self
```

**Summary**

Устанавливает amount.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  |  |

**Returns:** self - 


<a name="method_setConnectedAccountId" class="anchor"></a>
#### public setConnectedAccountId() : self

```php
public setConnectedAccountId(string|null $value = null) : self
```

**Summary**

Устанавливает идентификатор продавца.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Идентификатор продавца, подключенного к вашей платформе. |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $value = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Описание транзакции (не более 128 символов), которое продавец увидит в личном кабинете ЮKassa. Например: «Заказ маркетплейса №72». |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $value = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | value  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** self - 


<a name="method_setPlatformFeeAmount" class="anchor"></a>
#### public setPlatformFeeAmount() : self

```php
public setPlatformFeeAmount(\YooKassa\Model\AmountInterface|array|null $value = null) : self
```

**Summary**

Устанавливает platform_fee_amount.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  |  |

**Returns:** self - 


<a name="method_setReleaseFunds" class="anchor"></a>
#### public setReleaseFunds() : self

```php
public setReleaseFunds(bool|null $value = null) : self
```

**Summary**

Устанавливает порядок перевода денег продавцам.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | value  | Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать. |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $value = null) : self
```

**Summary**

Устанавливает status.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Transfer](../classes/YooKassa-Model-Payment-Transfer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  |  |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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