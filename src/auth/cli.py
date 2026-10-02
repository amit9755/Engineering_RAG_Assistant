"""Manage accounts from the command line (run from the project folder):

    python -m src.auth.cli list
    python -m src.auth.cli create <username> [--admin]
    python -m src.auth.cli reset-password <username>

Passwords are typed at a hidden prompt, never passed on the command line.
"""

import argparse
import getpass
import sys

from src.auth.store import user_store


def _ask_password() -> str:
    first = getpass.getpass("New password (min 8 characters): ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match")
    return first


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.auth.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    create = sub.add_parser("create")
    create.add_argument("username")
    create.add_argument("--admin", action="store_true")
    reset = sub.add_parser("reset-password")
    reset.add_argument("username")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for user in user_store.list_users():
                print(f"{user.username:20} {user.role:6} {'(must change password)' if user.must_change_password else ''}")
        elif args.command == "create":
            user_store.create_user(args.username, _ask_password(), "admin" if args.admin else "user")
            print(f"Created {args.username}")
        else:
            user_store.set_password(args.username, _ask_password())
            print(f"Password changed for {args.username}; their other sessions were signed out")
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
