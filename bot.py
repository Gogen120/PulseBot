import os
import random
import discord
from config import NUMBER_OF_ITERATIONS
from lib.text_generation import create_ode

client = discord.Client()


def get_members(server_members):
    members = [member.name for member in server_members]
    return '\n'.join(members)


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


@client.event
async def on_message(message):
    if message.content.startswith('!ode'):
        await client.send_message(message.channel, 'Creating ode...')
        ode = create_ode()
        await client.send_message(
            message.channel, f'Ode to Pulse №{random.randint(1, 10000)}:'
        )
        await client.send_message(message.channel, '-----------------------------------')
        await client.send_message(message.channel, ode)
        await client.send_message(message.channel, '-----------------------------------')
    elif message.content.startswith('!hi'):
        await client.send_message(
            message.channel,
            f'Hello {message.author.mention}. I am Pulse Bot! Use `!commands` to check all of my commands'
        )
    elif message.content.startswith('!commands'):
        await client.send_message(
            message.channel,
            f'List of the commands: ```!hi - intro command\n!ode - genereate ode to Pulse\n!info - information about Bot\n!home - Where do I live```'
        )
    elif message.content.startswith('!info'):
        await client.send_message(
            message.channel,
            f'These bot generate odes to Pulse, using neural network trained on Pushkin\'s poems (currently with {NUMBER_OF_ITERATIONS} iterations)'
        )
    elif message.content.startswith('!home'):
        await client.send_message(
            message.channel,
            f'Code for bot you can find here: https://github.com/Gogen120/PulseBot'
        )
    elif message.content.startswith('!members'):
        members = get_members(message.server.members)
        await client.send_message(
            message.channel,
            f'Channel members:\n{members}'
        )


if __name__ == '__main__':
    client.run(os.environ['DISCORD_TOKEN'])
