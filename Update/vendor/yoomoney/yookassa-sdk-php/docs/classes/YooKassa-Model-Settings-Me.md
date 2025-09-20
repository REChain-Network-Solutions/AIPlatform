# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Settings\Me
### Namespace: [\YooKassa\Model\Settings](../namespaces/yookassa-model-settings.md)
---
**Summary:**

Класс, представляющий модель Me.

**Description:**

Объект настроек магазина или шлюза (Me) — актуальная информация о настройках магазина или шлюза.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [STATUS_ENABLED](../classes/YooKassa-Model-Settings-Me.md#constant_STATUS_ENABLED) |  | Подключен к ЮKassa, может проводить платежи или выплаты |
| public | [STATUS_DISABLED](../classes/YooKassa-Model-Settings-Me.md#constant_STATUS_DISABLED) |  | Не может проводить платежи или выплаты (еще не подключен, закрыт или временно не работает) |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account_id](../classes/YooKassa-Model-Settings-Me.md#property_account_id) |  | Идентификатор магазина или шлюза. |
| public | [$accountId](../classes/YooKassa-Model-Settings-Me.md#property_accountId) |  | Идентификатор магазина или шлюза. |
| public | [$fiscalization](../classes/YooKassa-Model-Settings-Me.md#property_fiscalization) |  | Настройки магазина для [отправки чеков в налоговую](https://yookassa.ru/developers/payment-acceptance/receipts/basics). |
| public | [$fiscalization_enabled](../classes/YooKassa-Model-Settings-Me.md#property_fiscalization_enabled) |  | Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков. |
| public | [$fiscalizationEnabled](../classes/YooKassa-Model-Settings-Me.md#property_fiscalizationEnabled) |  | Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков. |
| public | [$itn](../classes/YooKassa-Model-Settings-Me.md#property_itn) |  | ИНН магазина (10 или 12 цифр). Присутствует, если вы запрашивали настройки магазина. |
| public | [$name](../classes/YooKassa-Model-Settings-Me.md#property_name) |  | Название шлюза, которое отображается в личном кабинете ЮKassa. Присутствует, если вы запрашивали настройки шлюза. |
| public | [$payment_methods](../classes/YooKassa-Model-Settings-Me.md#property_payment_methods) |  | Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина. |
| public | [$paymentMethods](../classes/YooKassa-Model-Settings-Me.md#property_paymentMethods) |  | Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина. |
| public | [$payout_balance](../classes/YooKassa-Model-Settings-Me.md#property_payout_balance) |  | Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза. |
| public | [$payout_methods](../classes/YooKassa-Model-Settings-Me.md#property_payout_methods) |  | Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза. |
| public | [$payoutBalance](../classes/YooKassa-Model-Settings-Me.md#property_payoutBalance) |  | Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза. |
| public | [$payoutMethods](../classes/YooKassa-Model-Settings-Me.md#property_payoutMethods) |  | Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза. |
| public | [$status](../classes/YooKassa-Model-Settings-Me.md#property_status) |  | Статус магазина или шлюза. |
| public | [$test](../classes/YooKassa-Model-Settings-Me.md#property_test) |  | Это тестовый магазин или шлюз. |

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
| public | [getAccountId()](../classes/YooKassa-Model-Settings-Me.md#method_getAccountId) |  | Возвращает идентификатор магазина или шлюза. |
| public | [getFiscalization()](../classes/YooKassa-Model-Settings-Me.md#method_getFiscalization) |  | Возвращает настройки магазина для отправки чеков в налоговую. |
| public | [getFiscalizationEnabled()](../classes/YooKassa-Model-Settings-Me.md#method_getFiscalizationEnabled) | *deprecated* | Возвращает признак включенной фискализации. |
| public | [getItn()](../classes/YooKassa-Model-Settings-Me.md#method_getItn) |  | Возвращает ИНН магазина. |
| public | [getName()](../classes/YooKassa-Model-Settings-Me.md#method_getName) |  | Возвращает название шлюза. |
| public | [getPaymentMethods()](../classes/YooKassa-Model-Settings-Me.md#method_getPaymentMethods) |  | Возвращает список способов оплаты, доступных магазину. |
| public | [getPayoutBalance()](../classes/YooKassa-Model-Settings-Me.md#method_getPayoutBalance) |  | Возвращает баланс вашего шлюза. |
| public | [getPayoutMethods()](../classes/YooKassa-Model-Settings-Me.md#method_getPayoutMethods) |  | Возвращает список способов получения выплат. |
| public | [getStatus()](../classes/YooKassa-Model-Settings-Me.md#method_getStatus) |  | Возвращает статус магазина или шлюза. |
| public | [getTest()](../classes/YooKassa-Model-Settings-Me.md#method_getTest) |  | Возвращает тестовый магазин или шлюз. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccountId()](../classes/YooKassa-Model-Settings-Me.md#method_setAccountId) |  | Устанавливает идентификатор магазина или шлюза. |
| public | [setFiscalization()](../classes/YooKassa-Model-Settings-Me.md#method_setFiscalization) |  | Устанавливает настройки магазина для отправки чеков в налоговую. |
| public | [setFiscalizationEnabled()](../classes/YooKassa-Model-Settings-Me.md#method_setFiscalizationEnabled) | *deprecated* | Устанавливает признак включенной фискализации. |
| public | [setItn()](../classes/YooKassa-Model-Settings-Me.md#method_setItn) |  | Устанавливает ИНН магазина. |
| public | [setName()](../classes/YooKassa-Model-Settings-Me.md#method_setName) |  | Устанавливает название шлюза. |
| public | [setPaymentMethods()](../classes/YooKassa-Model-Settings-Me.md#method_setPaymentMethods) |  | Устанавливает список способов оплаты, доступных магазину. |
| public | [setPayoutBalance()](../classes/YooKassa-Model-Settings-Me.md#method_setPayoutBalance) |  | Устанавливает Баланс вашего шлюза. |
| public | [setPayoutMethods()](../classes/YooKassa-Model-Settings-Me.md#method_setPayoutMethods) |  | Устанавливает список способов получения выплат. |
| public | [setStatus()](../classes/YooKassa-Model-Settings-Me.md#method_setStatus) |  | Устанавливает статус магазина или шлюза. |
| public | [setTest()](../classes/YooKassa-Model-Settings-Me.md#method_setTest) |  | Устанавливает тестовый магазин или шлюз. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Settings/Me.php](../../lib/Model/Settings/Me.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Settings\Me

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
<a name="constant_STATUS_ENABLED" class="anchor"></a>
###### STATUS_ENABLED
Подключен к ЮKassa, может проводить платежи или выплаты

```php
STATUS_ENABLED = 'enabled'
```


<a name="constant_STATUS_DISABLED" class="anchor"></a>
###### STATUS_DISABLED
Не может проводить платежи или выплаты (еще не подключен, закрыт или временно не работает)

```php
STATUS_DISABLED = 'disabled'
```



---
## Properties
<a name="property_account_id"></a>
#### public $account_id : string
---
***Description***

Идентификатор магазина или шлюза.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_accountId"></a>
#### public $accountId : string
---
***Description***

Идентификатор магазина или шлюза.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_fiscalization"></a>
#### public $fiscalization : \YooKassa\Model\Settings\FiscalizationData
---
***Description***

Настройки магазина для [отправки чеков в налоговую](https://yookassa.ru/developers/payment-acceptance/receipts/basics).

**Type:** <a href="../classes/YooKassa-Model-Settings-FiscalizationData.html"><abbr title="\YooKassa\Model\Settings\FiscalizationData">FiscalizationData</abbr></a>

**Details:**


<a name="property_fiscalization_enabled"></a>
#### public $fiscalization_enabled : bool
---
***Description***

Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_fiscalizationEnabled"></a>
#### public $fiscalizationEnabled : bool
---
***Description***

Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_itn"></a>
#### public $itn : string|null
---
***Description***

ИНН магазина (10 или 12 цифр). Присутствует, если вы запрашивали настройки магазина.

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_name"></a>
#### public $name : string|null
---
***Description***

Название шлюза, которое отображается в личном кабинете ЮKassa. Присутствует, если вы запрашивали настройки шлюза.

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_payment_methods"></a>
#### public $payment_methods : string[]|array
---
***Description***

Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина.

**Type:** <a href="../string[]|array"><abbr title="string[]|array">string[]|array</abbr></a>

**Details:**


<a name="property_paymentMethods"></a>
#### public $paymentMethods : string[]|array
---
***Description***

Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина.

**Type:** <a href="../string[]|array"><abbr title="string[]|array">string[]|array</abbr></a>

**Details:**


<a name="property_payout_balance"></a>
#### public $payout_balance : \YooKassa\Model\AmountInterface|null
---
***Description***

Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.

**Type:** <a href="../\YooKassa\Model\AmountInterface|null"><abbr title="\YooKassa\Model\AmountInterface|null">AmountInterface|null</abbr></a>

**Details:**


<a name="property_payout_methods"></a>
#### public $payout_methods : string[]|array
---
***Description***

Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза.

**Type:** <a href="../string[]|array"><abbr title="string[]|array">string[]|array</abbr></a>

**Details:**


<a name="property_payoutBalance"></a>
#### public $payoutBalance : \YooKassa\Model\AmountInterface|null
---
***Description***

Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.

**Type:** <a href="../\YooKassa\Model\AmountInterface|null"><abbr title="\YooKassa\Model\AmountInterface|null">AmountInterface|null</abbr></a>

**Details:**


<a name="property_payoutMethods"></a>
#### public $payoutMethods : string[]|array
---
***Description***

Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза.

**Type:** <a href="../string[]|array"><abbr title="string[]|array">string[]|array</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус магазина или шлюза.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_test"></a>
#### public $test : bool
---
***Description***

Это тестовый магазин или шлюз.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

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

Возвращает идентификатор магазина или шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string|null - 


<a name="method_getFiscalization" class="anchor"></a>
#### public getFiscalization() : \YooKassa\Model\Settings\FiscalizationData|null

```php
public getFiscalization() : \YooKassa\Model\Settings\FiscalizationData|null
```

**Summary**

Возвращает настройки магазина для отправки чеков в налоговую.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** \YooKassa\Model\Settings\FiscalizationData|null - 


<a name="method_getFiscalizationEnabled" class="anchor"></a>
#### (deprecated) - public getFiscalizationEnabled() : bool|null

```php
public getFiscalizationEnabled() : bool|null
```

**Summary**

Возвращает признак включенной фискализации.

**Deprecated**
DeprecatedУстарел.
**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** bool|null - 


<a name="method_getItn" class="anchor"></a>
#### public getItn() : string|null

```php
public getItn() : string|null
```

**Summary**

Возвращает ИНН магазина.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string|null - 


<a name="method_getName" class="anchor"></a>
#### public getName() : string|null

```php
public getName() : string|null
```

**Summary**

Возвращает название шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string|null - 


<a name="method_getPaymentMethods" class="anchor"></a>
#### public getPaymentMethods() : string[]|array|null

```php
public getPaymentMethods() : string[]|array|null
```

**Summary**

Возвращает список способов оплаты, доступных магазину.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string[]|array|null - 


<a name="method_getPayoutBalance" class="anchor"></a>
#### public getPayoutBalance() : \YooKassa\Model\AmountInterface|null

```php
public getPayoutBalance() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает баланс вашего шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_getPayoutMethods" class="anchor"></a>
#### public getPayoutMethods() : string[]|null

```php
public getPayoutMethods() : string[]|null
```

**Summary**

Возвращает список способов получения выплат.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string[]|null - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус магазина или шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** string|null - 


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool|null

```php
public getTest() : bool|null
```

**Summary**

Возвращает тестовый магазин или шлюз.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

**Returns:** bool|null - 


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
public setAccountId(string|int|null $account_id = null) : self
```

**Summary**

Устанавливает идентификатор магазина или шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR int OR null</code> | account_id  |  |

**Returns:** self - 


<a name="method_setFiscalization" class="anchor"></a>
#### public setFiscalization() : self

```php
public setFiscalization(\YooKassa\Model\Settings\FiscalizationData|array|null $fiscalization = null) : self
```

**Summary**

Устанавливает настройки магазина для отправки чеков в налоговую.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Settings\FiscalizationData OR array OR null</code> | fiscalization  | Настройки магазина для отправки чеков в налоговую. |

**Returns:** self - 


<a name="method_setFiscalizationEnabled" class="anchor"></a>
#### (deprecated) - public setFiscalizationEnabled() : self

```php
public setFiscalizationEnabled(bool|null $fiscalization_enabled = null) : self
```

**Summary**

Устанавливает признак включенной фискализации.

**Deprecated**
DeprecatedУстарел.
**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | fiscalization_enabled  | Признак включенной фискализации. |

**Returns:** self - 


<a name="method_setItn" class="anchor"></a>
#### public setItn() : self

```php
public setItn(string|null $itn = null) : self
```

**Summary**

Устанавливает ИНН магазина.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | itn  | ИНН магазина (10 или 12 цифр). |

**Returns:** self - 


<a name="method_setName" class="anchor"></a>
#### public setName() : self

```php
public setName(string|null $name = null) : self
```

**Summary**

Устанавливает название шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | name  | Название шлюза, которое отображается в личном кабинете ЮKassa. |

**Returns:** self - 


<a name="method_setPaymentMethods" class="anchor"></a>
#### public setPaymentMethods() : self

```php
public setPaymentMethods(string[]|array|null $payment_methods = null) : self
```

**Summary**

Устанавливает список способов оплаты, доступных магазину.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string[] OR array OR null</code> | payment_methods  | Список способов оплаты, доступных магазину. |

**Returns:** self - 


<a name="method_setPayoutBalance" class="anchor"></a>
#### public setPayoutBalance() : self

```php
public setPayoutBalance(\YooKassa\Model\AmountInterface|array|null $payout_balance = null) : self
```

**Summary**

Устанавливает Баланс вашего шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | payout_balance  | Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза. |

**Returns:** self - 


<a name="method_setPayoutMethods" class="anchor"></a>
#### public setPayoutMethods() : self

```php
public setPayoutMethods(string[]|array|null $payout_methods = null) : self
```

**Summary**

Устанавливает список способов получения выплат.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string[] OR array OR null</code> | payout_methods  | Список способов получения выплат, доступных шлюзу. |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает статус магазина или шлюза.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус магазина или шлюза. |

**Returns:** self - 


<a name="method_setTest" class="anchor"></a>
#### public setTest() : self

```php
public setTest(bool|null $test = null) : self
```

**Summary**

Устанавливает тестовый магазин или шлюз.

**Details:**
* Inherited From: [\YooKassa\Model\Settings\Me](../classes/YooKassa-Model-Settings-Me.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | test  | Это тестовый магазин или шлюз. |

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