<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\SelfEmployed;

use DateTime;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SelfEmployed.
 *
 * Объект самозанятого (SelfEmployed) — актуальная информация о самозанятом.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор самозанятого в ЮKassa.
 * @property string $status Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.
 * @property DateTime $created_at Время создания объекта самозанятого.
 * @property DateTime $createdAt Время создания объекта самозанятого.
 * @property null|string $itn ИНН самозанятого.
 * @property null|string $phone Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог.
 * @property null|SelfEmployedConfirmation $confirmation Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог.
 * @property bool $test Признак тестовой операции.
 */
class SelfEmployed extends AbstractObject implements SelfEmployedInterface
{
    /**
     * Идентификатор самозанятого в ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: SelfEmployedInterface::MAX_LENGTH_ID)]
    #[Assert\Length(min: SelfEmployedInterface::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [SelfEmployedStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = SelfEmployedStatus::PENDING;

    /**
     * Время создания объекта самозанятого.
     *
     * Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: ~`2017-11-03T11:52:31.827Z
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
     * ИНН самозанятого. Формат: 12 цифр без пробелов.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^\d{12}$/")]
    protected ?string $_itn = null;

    /**
     * Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{4,15}$/")]
    protected ?string $_phone = null;

    /**
     * Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог.
     *
     * @var SelfEmployedConfirmation|null
     */
    #[Assert\Type(SelfEmployedConfirmation::class)]
    protected ?SelfEmployedConfirmation $_confirmation = null;

    /**
     * Признак тестовой операции
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected ?bool $_test = false;

    /**
     * Возвращает идентификатор самозанятого в ЮKassa.
     *
     * @return string|null
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор самозанятого в ЮKassa.
     *
     * @param string|null $id Идентификатор самозанятого в ЮKassa.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.
     *
     * @return string|null
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.
     *
     * @param string|null $status Статус подключения самозанятого
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает время создания объекта самозанятого.
     *
     * @return DateTime|null Время создания объекта самозанятого
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания объекта самозанятого.
     *
     * @param DateTime|string|null $created_at Время создания объекта самозанятого.
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает ИНН самозанятого.
     *
     * @return string|null
     */
    public function getItn(): ?string
    {
        return $this->_itn;
    }

    /**
     * Устанавливает ИНН самозанятого.
     *
     * @param string|null $itn ИНН самозанятого
     *
     * @return self
     */
    public function setItn(?string $itn = null): self
    {
        $this->_itn = $this->validatePropertyValue('_itn', $itn);
        return $this;
    }

    /**
     * Возвращает телефон самозанятого.
     *
     * @return string|null Телефон самозанятого
     */
    public function getPhone(): ?string
    {
        return $this->_phone;
    }

    /**
     * Устанавливает телефон самозанятого.
     *
     * @param string|null $phone Телефон самозанятого
     *
     * @return self
     */
    public function setPhone(?string $phone = null): self
    {
        $this->_phone = $this->validatePropertyValue('_phone', $phone);
        return $this;
    }

    /**
     * Возвращает сценарий подтверждения.
     *
     * @return SelfEmployedConfirmation|null Сценарий подтверждения
     */
    public function getConfirmation(): ?SelfEmployedConfirmation
    {
        return $this->_confirmation;
    }

    /**
     * Устанавливает сценарий подтверждения.
     *
     * @param SelfEmployedConfirmation|array|null $confirmation Сценарий подтверждения
     *
     * @return $this
     */
    public function setConfirmation(mixed $confirmation = null): self
    {
        if (is_array($confirmation)) {
            $confirmation = (new SelfEmployedConfirmationFactory)->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool Признак тестовой операции
     */
    public function getTest(): bool
    {
        return $this->_test;
    }

    /**
     * Устанавливает признак тестовой операции.
     *
     * @param bool|null $test Признак тестовой операции
     *
     * @return self
     */
    public function setTest(?bool $test = null): self
    {
        $this->_test = $this->validatePropertyValue('_test', $test);
        return $this;
    }
}
